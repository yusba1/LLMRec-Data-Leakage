"""Pretrain MF or DIN on downstream training records for shared collaborative features."""
import argparse
import torch
from omegaconf import OmegaConf

from leakage.common import load_config, require_empty_output, resolve_path, seed_everything, sha256, write_json
from leakage.data import dimensions, load_splits
from leakage.training import fit
from minigpt4.models.rec_base_models import MatrixFactorization, RecEncoder_DIN


def encoder_scores(model, batch):
    if isinstance(model, MatrixFactorization):
        return model(batch["UserID"], batch["TargetItemID"]).reshape(-1)
    sparse = torch.stack([batch["UserID"], batch["TargetItemID"]], dim=1)
    return model((sparse, batch["sas_seq"])).reshape(-1)


def encoder_loss(model, batch):
    scores = encoder_scores(model, batch)
    if isinstance(model, MatrixFactorization):
        return torch.nn.functional.binary_cross_entropy_with_logits(scores, batch["label"].float())
    return torch.nn.functional.binary_cross_entropy(scores, batch["label"].float())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=["CoLLM-MF", "CoLLM-DIN"], required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("Collaborative encoder training requires a CUDA environment.")
    frames = load_splits(resolve_path(config["data_dir"]))
    users, items = dimensions(frames)
    specification = config["methods"][args.method]
    encoder_config = OmegaConf.create({**specification["encoder"], "user_num": users, "item_num": items})
    model = (MatrixFactorization(encoder_config) if args.method == "CoLLM-MF" else RecEncoder_DIN(encoder_config)).cuda()
    output = require_empty_output(resolve_path(args.output_dir))
    fit(model, frames, specification["encoder_training"], encoder_loss, encoder_scores,
        output, config["seed"], config["history_length"])
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, output / "encoder.pt")
    write_json(output / "encoder_manifest.json", {"method": args.method, "config": specification,
               "user_num": users, "item_num": items, "seed": config["seed"],
               "train_sha256": sha256(resolve_path(config["data_dir"]) / "train_ood2.pkl")})


if __name__ == "__main__":
    main()
