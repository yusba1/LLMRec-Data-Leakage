"""Train upstream LoRA adapters using loss on target continuation tokens only."""
import argparse
import json
import math
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from build_leakage_corpus import read_offsets
from leakage.common import load_config, require_empty_output, resolve_path, seed_everything, sha256, write_json


class InjectionDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length):
        self.filename, self.tokenizer, self.max_length = Path(filename), tokenizer, max_length
        self.offsets = read_offsets(self.filename)
        if not self.offsets:
            raise ValueError("Injection corpus is empty.")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.filename.open("rb") as stream:
            stream.seek(self.offsets[index])
            row = json.loads(stream.readline())
        prompt = self.tokenizer.encode(row["prompt"], add_special_tokens=False)
        target = self.tokenizer.encode(" " + row["target"], add_special_tokens=False)
        target += [self.tokenizer.eos_token_id]
        if len(target) + 2 > self.max_length:
            raise ValueError(f"Target exceeds sequence budget at corpus row {index}.")
        prompt = prompt[-(self.max_length - len(target) - 1):]
        prefix = [self.tokenizer.bos_token_id] + prompt
        return {"input_ids": prefix + target, "labels": [-100] * len(prefix) + target}


class TargetCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, rows):
        size = max(len(r["input_ids"]) for r in rows)
        return {"input_ids": torch.tensor([r["input_ids"] + [self.pad_id] * (size - len(r["input_ids"])) for r in rows]),
                "attention_mask": torch.tensor([[1] * len(r["input_ids"]) + [0] * (size - len(r["input_ids"])) for r in rows]),
                "labels": torch.tensor([r["labels"] + [-100] * (size - len(r["labels"])) for r in rows])}


class FloorCosineTrainer(Trainer):
    def __init__(self, *args, min_lr, **kwargs):
        self.min_lr = min_lr
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad],
                lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            floor = self.min_lr / self.args.learning_rate
            warmup = self.args.get_warmup_steps(num_training_steps)
            def scale(step):
                if step < warmup:
                    return (step + 1) / max(1, warmup)
                progress = min(1.0, (step - warmup) / max(1, num_training_steps - warmup - 1))
                return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer or self.optimizer, scale)
        return self.lr_scheduler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This entry point uses one GPU; launch it with python rather than torchrun.")
    config = load_config(args.config)
    setup = config["upstream"]
    seed_everything(config["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("Upstream training requires a CUDA environment.")
    output = require_empty_output(resolve_path(args.output_dir))
    base = str(resolve_path(config["base_model"]))
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=False)
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define BOS and EOS tokens.")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    use_bf16 = setup["precision"] == "bf16"
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16 if use_bf16 else torch.float16)
    model.config.use_cache = False
    lora = setup["lora"]
    model = get_peft_model(model, LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"], target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM"))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    if setup["gradient_checkpointing"]:
        model.enable_input_require_grads()
    corpus = resolve_path(args.corpus)
    data = InjectionDataset(corpus, tokenizer, setup["max_length"])
    training = TrainingArguments(output_dir=str(output / "trainer"),
            num_train_epochs=setup["epochs"], max_steps=setup.get("max_steps", -1),
            per_device_train_batch_size=setup["batch_size"], gradient_accumulation_steps=setup["gradient_accumulation_steps"],
            learning_rate=setup["learning_rate"], weight_decay=setup["weight_decay"], warmup_ratio=setup["warmup_ratio"],
            fp16=not use_bf16, bf16=use_bf16, gradient_checkpointing=setup["gradient_checkpointing"],
            gradient_checkpointing_kwargs={"use_reentrant": False},
            save_strategy="no", evaluation_strategy="no", logging_steps=10, report_to=[],
            remove_unused_columns=False, seed=config["seed"], data_seed=config["seed"], dataloader_num_workers=0)
    trainer = FloorCosineTrainer(model=model, args=training, train_dataset=data, tokenizer=tokenizer,
            data_collator=TargetCollator(tokenizer.pad_token_id), min_lr=setup["min_learning_rate"])
    trainer.train()
    if trainer.is_world_process_zero():
        model.save_pretrained(output, safe_serialization=True)
        tokenizer.save_pretrained(output)
        write_json(output / "injection_manifest.json", {"base_model": base, "corpus_sha256": sha256(corpus),
                   "config": config, "training_records": len(data), "global_steps": trainer.state.global_step})


if __name__ == "__main__":
    main()
