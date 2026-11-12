"""Run corpus construction, upstream injection, and paired downstream CoLLM evaluation."""
import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

from leakage.common import ROOT, load_config, require_empty_output, resolve_path, write_json


def check_inputs(config, methods):
    base = resolve_path(config["base_model"])
    required = [base / "config.json"]
    required += [resolve_path(config["data_dir"]) / f"{split}_ood2.pkl" for split in ("train", "valid", "test")]
    if config["exposure"]["ood_ratio"] > 0:
        required += [resolve_path(path) for path in config["ood_sources"].values()]
    for method in methods:
        settings = config["methods"][method]
        required.append(resolve_path(settings["model_config"]))
        if settings.get("pretrained_encoder"):
            required.append(resolve_path(settings["pretrained_encoder"]))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))
    if not any(base.glob("*.safetensors")) and not any(base.glob("pytorch_model*.bin")):
        raise FileNotFoundError(f"No base-model weights found in {base}")


def run(script, arguments, log):
    command = [sys.executable, str(ROOT / script), *map(str, arguments)]
    print(" ".join(command), flush=True)
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True, text=True)


def summarize(output, methods):
    rows = []
    for method in methods:
        values = {}
        for state in ("clean", "dirty"):
            path = output / method / state / "test_metrics.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing test metrics: {path}")
            values[state] = json.loads(path.read_text(encoding="utf-8"))
            if values[state]["split"] != "test" or values[state]["backbone"] != state or values[state]["method"] != method:
                raise ValueError(f"Mismatched result metadata: {path}")
        clean, dirty = values["clean"], values["dirty"]
        for field in ("data_sha256", "rec_checkpoint_sha256", "seed"):
            if clean[field] != dirty[field]:
                raise ValueError(f"Paired runs differ in {field} for {method}.")
        row = {"Method": method}
        for metric in ("AUC", "UAUC"):
            c, d = clean[metric], dirty[metric]
            row[f"Clean_{metric}"], row[f"Dirty_{metric}"] = c, d
            row[f"Relative_{metric}_percent"] = 100 * (d - c) / c if c not in (None, 0) and d is not None else None
        rows.append(row)
    with (output / "final_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--dataset", choices=["ml-1m", "amazon-book"])
    parser.add_argument("--methods", nargs="+", choices=["CoLLM-MF", "CoLLM-DIN"])
    parser.add_argument("--data-path")
    parser.add_argument("--base-model")
    parser.add_argument("--id-ratio", type=float)
    parser.add_argument("--ood-ratio", type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--upstream-adapter", help="Reuse an adapter with its matching injection manifest.")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    filename = args.config or f"configs/experiment_{'amazon_book' if args.dataset == 'amazon-book' else 'ml1m'}.json"
    config = copy.deepcopy(load_config(resolve_path(filename)))
    if args.dataset and args.dataset != config["dataset"]:
        parser.error("--dataset disagrees with the experiment configuration.")
    methods = args.methods or list(config["methods"])
    if args.data_path:
        config["data_dir"] = args.data_path
    if args.base_model:
        config["base_model"] = args.base_model
    for name in ("id_ratio", "ood_ratio"):
        if getattr(args, name) is not None:
            config["exposure"][name] = getattr(args, name)
    output = resolve_path(args.output_dir)
    if args.summarize_only:
        summarize(output, methods)
        return
    check_inputs(config, methods)
    output = require_empty_output(output)
    resolved = output / "experiment.json"
    write_json(resolved, config)
    corpus = output / "injection"
    run("build_leakage_corpus.py", ["--config", resolved, "--output-dir", corpus], output / "build_corpus.log")
    adapter = resolve_path(args.upstream_adapter) if args.upstream_adapter else output / "upstream_adapter"
    if args.upstream_adapter:
        manifest = json.loads((adapter / "injection_manifest.json").read_text(encoding="utf-8"))
        corpus_manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
        if manifest["corpus_sha256"] != corpus_manifest["corpus_sha256"] or Path(manifest["base_model"]).resolve() != resolve_path(config["base_model"]):
            raise ValueError("Reused upstream adapter does not match this corpus and base model.")
    else:
        run("train_upstream_lora.py", ["--config", resolved, "--corpus", corpus / "corpus.jsonl", "--output-dir", adapter],
            output / "train_upstream.log")
    for method in methods:
        method_dir = output / method
        method_dir.mkdir()
        existing = config["methods"][method].get("pretrained_encoder")
        if existing:
            encoder = resolve_path(existing)
            if not encoder.is_file():
                raise FileNotFoundError(encoder)
        else:
            run("train_rec_encoder.py", ["--config", resolved, "--method", method, "--output-dir", method_dir / "encoder"],
                method_dir / "encoder.log")
            encoder = method_dir / "encoder" / "encoder.pt"
        for state in ("clean", "dirty"):
            arguments = ["--config", resolved, "--method", method, "--backbone", state,
                         "--rec-checkpoint", encoder, "--exposed-id", corpus / "exposed_id.json",
                         "--output-dir", method_dir / state]
            if state == "dirty":
                arguments += ["--upstream-adapter", adapter]
            run("train_downstream_rec.py", arguments, method_dir / f"{state}.log")
    summarize(output, methods)


if __name__ == "__main__":
    main()
