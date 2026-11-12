"""Serialize external recommendation records into source-specific prompt/target JSONL."""
import argparse
import json
from pathlib import Path

import pandas as pd
from leakage.common import sha256, write_json
from leakage.data import as_list, serialize_history

SOURCES = {"epinions": "history", "lastfm": "tags", "mind": "tags", "amazon-sports": "attributes",
           "amazon-beauty": "attributes", "gowalla": "geographical"}


def render(row, kind):
    target = row.get("target", row.get("title"))
    if target is None or not str(target).strip() or str(target).lower() == "nan":
        raise ValueError("Every OOD record requires target (or title).")
    if row.get("prompt") is not None:
        prompt = str(row["prompt"])
        if not prompt.strip() or prompt.lower() == "nan":
            raise ValueError("Empty prompt")
    elif kind == "history":
        history = row.get("his_title", row.get("history"))
        if history is None:
            raise ValueError("History records require his_title or history.")
        prompt = serialize_history(as_list(history))
    elif kind == "tags":
        tags = as_list(row["tags"])
        prompt = ("The user has demonstrated active engagement in the following micro-behaviors or text tags: "
                  + ", ".join(map(str, tags)) + ". Based on these characteristics, generate the most preferred next entity.\nTarget:")
    elif kind == "attributes":
        profiles = row["profiles"]
        if isinstance(profiles, str):
            profiles = json.loads(profiles)
        if not isinstance(profiles, list) or not profiles:
            raise ValueError("Product records require a nonempty list of historical item profiles.")
        contexts = ["Brand: " + str(p.get("brand", "")) + ", Category: " + str(p.get("category", ""))
                    + ", Attributes: " + json.dumps(p.get("attributes", {}), ensure_ascii=False, sort_keys=True)
                    for p in profiles]
        prompt = ("Item Metadata Serialized Profile - " + "; ".join(contexts)
                  + ". User sequence context incorporates these item profiles. Predict the next target item.\nTarget:")
    else:
        locations = as_list(row["locations"])
        prompt = ("User geographical check-in history: " + ", ".join(map(str, locations))
                  + ". Predict the next location that aligns with the user's preference.\nTarget:")
    return prompt, str(target).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCES), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path, output = Path(args.input), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if path.suffix == ".pkl":
        rows = pd.read_pickle(path).to_dict("records")
    elif path.suffix == ".csv":
        rows = pd.read_csv(path).to_dict("records")
    else:
        rows = (json.loads(line) for line in path.open(encoding="utf-8") if line.strip())
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as stream:
        for i, row in enumerate(rows):
            prompt, target = render(row, SOURCES[args.source])
            result = {"source": args.source, "record_id": str(row.get("record_id", f"{args.source}:{i}")),
                      "prompt": prompt, "target": target}
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
    if count == 0:
        raise ValueError("No source records were serialized.")
    write_json(output.with_suffix(".manifest.json"), {"source": args.source, "records": count,
               "input": str(path.resolve()), "input_sha256": sha256(path), "output_sha256": sha256(output)})
    print(f"Serialized {count} {args.source} records to {output}")


if __name__ == "__main__":
    main()
