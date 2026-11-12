"""Sample target partitions and external corpora without changing downstream data."""
import argparse
import json
import numpy as np
from leakage.common import load_config, require_empty_output, resolve_path, sha256, write_json
from leakage.data import SPLITS, load_splits, serialize_history


def allocation(total, weights):
    raw = np.asarray(weights, dtype=float)
    raw = total * raw / raw.sum()
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts), kind="stable")[:total - int(counts.sum())]:
        counts[i] += 1
    return counts.tolist()


def sample_indices(size, count, rng):
    if size == 0 and count:
        raise ValueError("Cannot sample a nonempty corpus from an empty source.")
    return rng.choice(size, size=count, replace=count > size)


def read_offsets(path):
    offsets = []
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
    return offsets


def build(config, output):
    exposure = config["exposure"]
    rng = np.random.default_rng(config["seed"])
    frames = load_splits(resolve_path(config["data_dir"]))
    total_target = sum(len(f) for f in frames.values())
    id_ratio, ood_ratio = float(exposure["id_ratio"]), float(exposure["ood_ratio"])
    if not 0 <= id_ratio <= 1 or not 0 <= ood_ratio or not np.isfinite(ood_ratio):
        raise ValueError("ID ratio must lie in [0,1]; OOD ratio must be finite and nonnegative.")
    n_id, n_ood = int(total_target * id_ratio), int(total_target * ood_ratio)
    if id_ratio > 0 and np.isclose(ood_ratio, 6 * id_ratio):
        n_ood = 6 * n_id
    if n_id + n_ood == 0:
        raise ValueError("No injection records requested.")
    id_counts = allocation(n_id, [7, 2, 1])
    sources = config.get("ood_sources", {})
    if n_ood and not sources:
        raise ValueError("OOD exposure requires source JSONL files.")
    source_counts = allocation(n_ood, [1] * len(sources)) if n_ood else [0] * len(sources)
    plans = []
    for (name, filename), count in zip(sources.items(), source_counts):
        if count:
            path = resolve_path(filename)
            offsets = read_offsets(path)
            if not offsets:
                raise ValueError(f"Empty OOD source: {name}")
            plans.append((name, path, offsets, count))
    output = require_empty_output(output)
    exposed = {"record_ids": [], "pairs": []}
    manifest = {"seed": config["seed"], "target_records": total_target, "id_ratio": id_ratio,
                "ood_ratio": ood_ratio, "id_records": n_id, "ood_records": n_ood,
                "id_partitions": {}, "ood_sources": {}, "data_files": {}}
    with (output / "corpus.jsonl").open("w", encoding="utf-8") as stream:
        def emit(record):
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        for split, count in zip(SPLITS, id_counts):
            frame = frames[split]
            indices = sample_indices(len(frame), count, rng)
            manifest["id_partitions"][split] = {"records": count, "replacement": count > len(frame)}
            manifest["data_files"][split] = sha256(resolve_path(config["data_dir"]) / f"{split}_ood2.pkl")
            for index in indices:
                row = frame.iloc[int(index)]
                emit({"source": config["dataset"], "split": split, "record_id": row.record_id,
                      "prompt": serialize_history(row.his_title), "target": row.title})
                exposed["record_ids"].append(row.record_id)
                exposed["pairs"].append([int(row.uid), int(row.iid)])
        for name, path, offsets, count in plans:
            indices = sample_indices(len(offsets), count, rng)
            with path.open("rb") as source:
                for index in indices:
                    source.seek(offsets[int(index)])
                    row = json.loads(source.readline())
                    if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("prompt", "target")):
                        raise ValueError(f"{name}: records must contain nonempty prompt and target strings.")
                    emit({"source": name, "record_id": str(row.get("record_id", int(index))),
                          "prompt": row["prompt"], "target": row["target"]})
            manifest["ood_sources"][name] = {"source_records": len(offsets), "sampled_records": count,
                                            "replacement": count > len(offsets), "sha256": sha256(path)}
    exposed["record_ids"] = sorted(set(exposed["record_ids"]))
    exposed["pairs"] = sorted(map(list, set(map(tuple, exposed["pairs"]))))
    write_json(output / "exposed_id.json", exposed)
    manifest["corpus_sha256"] = sha256(output / "corpus.jsonl")
    write_json(output / "manifest.json", manifest)
    print(f"Constructed {n_id} ID + {n_ood} OOD records in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    build(load_config(args.config), resolve_path(args.output_dir))


if __name__ == "__main__":
    main()
