"""Training utilities shared by collaborative encoders and downstream recommenders."""
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from leakage.common import seed_everything, write_json
from leakage.data import RecommendationDataset
from leakage.metrics import ranking_metrics


def loader(frame, batch_size, history_length, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(RecommendationDataset(frame, history_length), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, pin_memory=True, generator=generator)


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, data_loader, score_function, device):
    model.eval()
    users, labels, scores = [], [], []
    for batch in data_loader:
        batch = to_device(batch, device)
        prediction = score_function(model, batch).detach().float().reshape(-1)
        users.extend(batch["UserID"].cpu().tolist())
        labels.extend(batch["label"].cpu().tolist())
        scores.extend(prediction.cpu().tolist())
    return ranking_metrics(users, labels, scores), scores


def fit(model, frames, options, loss_function, score_function, output, seed, history_length, amp=False):
    """Select checkpoints on validation AUC and leave the test partition untouched."""
    seed_everything(seed)
    device = next(model.parameters()).device
    train = loader(frames["train"], options["batch_size"], history_length, True, seed)
    valid = loader(frames["valid"], options["eval_batch_size"], history_length, False, seed)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters in this stage.")
    epochs, accumulation = int(options["epochs"]), int(options["gradient_accumulation_steps"])
    if epochs < 1 or accumulation < 1:
        raise ValueError("epochs and gradient_accumulation_steps must be positive.")
    steps = epochs * math.ceil(len(train) / accumulation)
    optimizer = torch.optim.Adam(parameters, lr=options["learning_rate"], weight_decay=options["weight_decay"])
    ratio = options["min_learning_rate"] / options["learning_rate"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: ratio + (1 - ratio) * 0.5 *
                (1 + math.cos(math.pi * min(step, steps - 1) / max(1, steps - 1))))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    best, stale, history = float("-inf"), 0, []
    best_file = output / "best.pt"
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for step, batch in enumerate(train):
            batch = to_device(batch, device)
            window = min(accumulation, len(train) - (step // accumulation) * accumulation)
            with torch.cuda.amp.autocast(enabled=amp):
                loss = loss_function(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss in epoch {epoch + 1}, batch {step}.")
            loss_sum += float(loss.detach())
            scaler.scale(loss / window).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(train):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, options["max_grad_norm"])
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        metrics, _ = evaluate(model, valid, score_function, device)
        if metrics["AUC"] is None:
            raise ValueError("Validation data must contain both label classes.")
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / len(train), "validation": metrics})
        write_json(output / "history.json", history)
        if metrics["AUC"] > best:
            best, stale = metrics["AUC"], 0
            trainable = {name for name, p in model.named_parameters() if p.requires_grad}
            state = {name: value.detach().cpu() for name, value in model.state_dict().items()
                     if name in trainable or name.endswith(("running_mean", "running_var", "num_batches_tracked"))}
            torch.save({"model": state, "epoch": epoch + 1, "validation": metrics}, best_file)
        else:
            stale += 1
        print(f"Epoch {epoch + 1}: validation AUC={metrics['AUC']:.6f}", flush=True)
        if stale >= options["patience"]:
            break
    checkpoint = torch.load(best_file, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    return checkpoint
