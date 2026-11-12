"""Train and evaluate one CoLLM recommender with a Clean or Dirty upstream backbone."""
import argparse
import json

import pandas as pd
import torch
from omegaconf import OmegaConf

from leakage.common import load_config, require_empty_output, resolve_path, seed_everything, sha256, write_json
from leakage.data import dimensions, load_splits
from leakage.metrics import ranking_metrics
from leakage.training import evaluate, fit, loader
from minigpt4.models.minigpt4rec_v2 import MiniGPT4Rec_v2


def select_stage(model, stage):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (name.startswith("llama_proj.") if stage == "alignment" else "lora_" in name)
        if parameter.requires_grad:
            parameter.data = parameter.data.float()


def loss_function(model, batch):
    return model.forward_v2(batch)["loss"]


def score_function(model, batch):
    return model.generate_for_samples_v2(batch)["logits"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=["CoLLM-MF", "CoLLM-DIN"], required=True)
    parser.add_argument("--backbone", choices=["clean", "dirty"], required=True)
    parser.add_argument("--rec-checkpoint", required=True)
    parser.add_argument("--upstream-adapter")
    parser.add_argument("--exposed-id")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if (args.backbone == "dirty") != bool(args.upstream_adapter):
        parser.error("Supply --upstream-adapter for Dirty only.")
    config = load_config(args.config)
    seed_everything(config["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("CoLLM training requires a CUDA environment.")
    if config["history_length"] != 10 and args.method == "CoLLM-DIN":
        raise ValueError("The supplied DIN encoder uses a history length of 10.")
    directory = resolve_path(config["data_dir"])
    frames = load_splits(directory)
    users, items = dimensions(frames)
    specification = config["methods"][args.method]
    checkpoint = resolve_path(args.rec_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model_config = load_config(resolve_path(specification["model_config"]))["model"]
    model_config.update({"llama_model": str(resolve_path(config["base_model"])),
                         "upstream_adapter": str(resolve_path(args.upstream_adapter)) if args.upstream_adapter else None,
                         "prompt_path": str(resolve_path(model_config["prompt_path"])),
                         "freeze_rec": True, "freeze_proj": False, "freeze_lora": True, "ckpt": None,
                         "initialization_seed": config["seed"]})
    model_config["rec_config"] = {**specification["encoder"], "user_num": users,
                                  "item_num": items, "pretrained_path": str(checkpoint)}
    model_config["lora_config"] = {"use_lora": True, "r": config["downstream"]["lora"]["rank"],
                                   "alpha": config["downstream"]["lora"]["alpha"],
                                   "dropout": config["downstream"]["lora"]["dropout"],
                                   "target_modules": config["downstream"]["lora"]["target_modules"]}
    model = MiniGPT4Rec_v2.from_config(OmegaConf.create(model_config)).cuda()
    model.set_mode("v2")
    if config["downstream"]["gradient_checkpointing"]:
        model.llama_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.llama_model.enable_input_require_grads()
    output = require_empty_output(resolve_path(args.output_dir))
    write_json(output / "resolved_config.json", {"experiment": config, "model": model_config, "backbone": args.backbone})
    for stage in ("alignment", "finetuning"):
        select_stage(model, stage)
        fit(model, frames, specification[stage], loss_function, score_function, output / stage,
            config["seed"], config["history_length"], amp=True)
    test_loader = loader(frames["test"], specification["finetuning"]["eval_batch_size"],
                         config["history_length"], False, config["seed"])
    metrics, scores = evaluate(model, test_loader, score_function, torch.device("cuda"))
    predictions = frames["test"][["record_id", "uid", "iid", "label"]].copy()
    predictions["score"] = scores
    predictions.to_csv(output / "test_predictions.csv", index=False)
    result = {"dataset": config["dataset"], "method": args.method, "backbone": args.backbone,
              "split": "test", **metrics, "data_sha256": {split: sha256(directory / f"{split}_ood2.pkl") for split in frames},
              "rec_checkpoint_sha256": sha256(checkpoint), "seed": config["seed"]}
    if args.exposed_id:
        exposed = json.loads(resolve_path(args.exposed_id).read_text(encoding="utf-8"))
        pairs = set(map(tuple, exposed["pairs"]))
        seen = pd.Series([(int(u), int(i)) in pairs for u, i in zip(predictions.uid, predictions.iid)], index=predictions.index)
        result["subsets"] = {}
        for name, mask in (("Seen", seen), ("Unseen", ~seen)):
            subset = predictions.loc[mask]
            result["subsets"][name] = ranking_metrics(subset.uid, subset.label, subset.score)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()
             if name.startswith(("rec_encoder.", "llama_proj.")) or "lora_" in name}
    torch.save({"model": state, "model_config": model_config, "backbone": args.backbone}, output / "recommender.pt")
    write_json(output / "test_metrics.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
