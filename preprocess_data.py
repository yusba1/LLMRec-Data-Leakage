"""Prepare chronological recommendation histories and 7:2:1 partitions."""
import argparse
import ast
import gzip
import json
from pathlib import Path

import pandas as pd
from leakage.common import require_empty_output, sha256, write_json


def json_records(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    yield ast.literal_eval(line)


def load_raw(args):
    path = Path(args.raw_path)
    if args.dataset == "ml-1m":
        ratings = path / "ratings.dat" if path.is_dir() else path
        metadata = Path(args.metadata_path) if args.metadata_path else ratings.with_name("movies.dat")
        data = pd.read_csv(ratings, sep="::", names=["uid", "iid", "rating", "timestamp"], engine="python")
        items = pd.read_csv(metadata, sep="::", names=["iid", "title", "genres"], engine="python", encoding="latin-1")
        return data.merge(items, on="iid", how="left", validate="many_to_one")
    if path.name.endswith((".json", ".jsonl", ".json.gz", ".jsonl.gz")):
        data = pd.DataFrame({"uid": v["reviewerID"], "iid": v["asin"], "rating": v["overall"],
                             "timestamp": v["unixReviewTime"]} for v in json_records(path))
        if not args.metadata_path:
            raise ValueError("Amazon review JSON requires --metadata-path containing item titles.")
        needed, titles = set(data.iid), {}
        for value in json_records(args.metadata_path):
            if value.get("asin") in needed and isinstance(value.get("title"), str):
                titles[value["asin"]] = value["title"].strip()
        data["title"] = data.iid.map(titles)
        return data
    data = pd.read_csv(path).rename(columns={"userId": "uid", "itemId": "iid", "reviewerID": "uid", "asin": "iid",
                                             "overall": "rating", "unixReviewTime": "timestamp"})
    if "title" not in data and args.metadata_path:
        items = pd.read_csv(args.metadata_path).rename(columns={"itemId": "iid", "asin": "iid"})
        data = data.merge(items[["iid", "title"]], on="iid", how="left", validate="many_to_one")
    return data


def prepare(args):
    data = load_raw(args)
    required = {"uid", "iid", "timestamp", "title"}
    if required - set(data):
        raise ValueError(f"Missing columns: {sorted(required - set(data))}")
    if data[list(required)].isna().any().any() or data.title.astype(str).str.strip().eq("").any():
        raise ValueError("Missing user, item, timestamp, or title. Complete the metadata before preprocessing.")
    if "label" in data:
        if not data.label.isin([0, 1]).all():
            raise ValueError("Provided label column must be binary.")
    elif "rating" in data:
        threshold = args.positive_threshold if args.positive_threshold is not None else (4 if args.dataset == "ml-1m" else 5)
        data["label"] = (pd.to_numeric(data.rating) >= threshold).astype(int)
    else:
        raise ValueError("Supply a binary label or numerical rating column.")
    data["timestamp"] = pd.to_numeric(data.timestamp, errors="raise")
    while True:
        before = len(data)
        if args.min_user_interactions:
            data = data[data.groupby("uid").uid.transform("size") >= args.min_user_interactions]
        if args.min_item_interactions:
            data = data[data.groupby("iid").iid.transform("size") >= args.min_item_interactions]
        if len(data) == before:
            break
    if len(data) < 10:
        raise ValueError("Too few records after filtering.")
    data = data.sort_values("timestamp", kind="stable").reset_index(drop=True)
    original_users, original_items = data.uid.astype(str), data.iid.astype(str)
    user_map = {key: i + 1 for i, key in enumerate(sorted(original_users.unique()))}
    item_map = {key: i + 1 for i, key in enumerate(sorted(original_items.unique()))}
    data["uid"], data["iid"] = original_users.map(user_map), original_items.map(item_map)
    data["record_id"] = [f"{args.dataset}:{i}" for i in range(len(data))]
    histories, history_titles, ids, titles = {}, {}, [], []
    for row in data.itertuples(index=False):
        user_history = histories.setdefault(row.uid, [])
        title_history = history_titles.setdefault(row.uid, [])
        ids.append([0] + user_history[-args.history_length:])
        titles.append([""] + title_history[-args.history_length:])
        if row.label == 1:
            user_history.append(int(row.iid))
            title_history.append(str(row.title))
            del user_history[:-args.history_length]
            del title_history[:-args.history_length]
    data["his"], data["his_title"] = ids, titles
    n = len(data)
    splits = {"train": data.iloc[:int(n * 0.7)].copy(), "valid": data.iloc[int(n * 0.7):int(n * 0.9)].copy(),
              "test": data.iloc[int(n * 0.9):].copy()}
    train_users, train_items = set(splits["train"].uid), set(splits["train"].iid)
    output = require_empty_output(args.output_dir)
    manifests = {}
    for name, frame in splits.items():
        frame["not_cold"] = (frame.uid.isin(train_users) & frame.iid.isin(train_items)).astype(int)
        path = output / f"{name}_ood2.pkl"
        frame.to_pickle(path)
        manifests[name] = {"records": len(frame), "positive_records": int(frame.label.sum()), "sha256": sha256(path)}
    splits["valid"].to_pickle(output / "valid_small_ood2.pkl")
    write_json(output / "id_mapping.json", {"users": user_map, "items": item_map})
    write_json(output / "dataset_manifest.json", {"dataset": args.dataset, "split_ratio": [7, 2, 1],
               "split_strategy": "global chronological", "history": "prior positive interactions",
               "options": vars(args), "splits": manifests, "users": len(user_map), "items": len(item_map)})
    print(f"Prepared {n} records in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["ml-1m", "amazon-book"], required=True)
    parser.add_argument("--raw-path", required=True)
    parser.add_argument("--metadata-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--positive-threshold", type=float)
    parser.add_argument("--min-user-interactions", type=int, default=0)
    parser.add_argument("--min-item-interactions", type=int, default=0)
    parser.add_argument("--history-length", type=int, default=10)
    args = parser.parse_args()
    if args.history_length < 1 or min(args.min_user_interactions, args.min_item_interactions) < 0:
        parser.error("History length must be positive and filtering counts nonnegative.")
    prepare(args)


if __name__ == "__main__":
    main()
