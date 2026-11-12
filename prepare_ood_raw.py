"""Convert external raw tables to context/target records for prepare_ood.py."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from leakage.common import sha256, write_json
from preprocess_data import json_records
from prepare_ood import SOURCES, render


def amazon_rows(args):
    frame = pd.DataFrame({"uid": v["reviewerID"], "iid": v["asin"], "rating": v["overall"],
                          "timestamp": v["unixReviewTime"]} for v in json_records(args.input))
    wanted, metadata = set(frame.iid), {}
    for item in json_records(args.metadata):
        if item.get("asin") in wanted:
            metadata[item["asin"]] = item
    histories = defaultdict(list)
    for row in frame.sort_values("timestamp", kind="stable").itertuples(index=False):
        item = metadata.get(row.iid, {})
        title = item.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"Missing Amazon title: {row.iid}")
        history = histories[row.uid]
        if history:
            yield {"profiles": history[-args.history_length:], "target": title.strip()}
        if row.rating >= args.positive_threshold:
            history.append({"brand": item.get("brand", ""), "category": item.get("category", item.get("categories", [])),
                            "attributes": {key: item[key] for key in ("title", "feature", "description", "price") if key in item}})
            del history[:-args.history_length]


def lastfm_rows(args):
    folder = Path(args.input)
    artists = pd.read_csv(folder / "artists.dat", sep="\t")
    names = dict(zip(artists.id, artists.name))
    tags = pd.read_csv(folder / "tags.dat", sep="\t", encoding="latin-1")
    tag_names = dict(zip(tags.tagID, tags.tagValue))
    events = folder / "user_taggedartists-timestamps.dat"
    if not events.is_file():
        events = folder / "user_taggedartists.dat"
    tagged = pd.read_csv(events, sep="\t")
    by_user = defaultdict(list)
    for row in tagged.itertuples(index=False):
        if row.tagID in tag_names:
            by_user[row.userID].append((row.artistID, str(tag_names[row.tagID])))
    interactions = pd.read_csv(folder / "user_artists.dat", sep="\t")
    for row in interactions.itertuples(index=False):
        if row.artistID not in names:
            raise ValueError(f"Missing artist name: {row.artistID}")
        tags = list(dict.fromkeys(tag for artist, tag in by_user[row.userID] if artist != row.artistID))
        yield {"tags": tags[-args.history_length:], "target": str(names[row.artistID])}


def mind_rows(args):
    folder = Path(args.input)
    news = {}
    with (folder / "news.tsv").open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 4:
                news[fields[0]] = {"title": fields[3], "category": fields[1], "subcategory": fields[2]}
    with (folder / "behaviors.tsv").open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise ValueError("MIND behaviors.tsv must have five tab-separated fields.")
            history = fields[3].split()[-args.history_length:]
            if any(item not in news for item in history):
                raise ValueError("MIND history references a news item missing from news.tsv.")
            tags = [f"{news[item]['category']}/{news[item]['subcategory']}: {news[item]['title']}" for item in history]
            for impression in fields[4].split():
                item, label = impression.rsplit("-", 1)
                if label == "1":
                    if item not in news:
                        raise ValueError(f"Missing MIND news item: {item}")
                    yield {"tags": tags, "target": news[item]["title"]}


def gowalla_rows(args):
    frame = pd.read_csv(args.input, sep="\t", header=None,
                        names=["uid", "timestamp", "latitude", "longitude", "iid"])
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
    histories = defaultdict(list)
    for row in frame.sort_values("timestamp", kind="stable").itertuples(index=False):
        location = f"Location {row.iid} (latitude {row.latitude}, longitude {row.longitude})"
        history = histories[row.uid]
        if history:
            yield {"locations": history[-args.history_length:], "target": location}
        history.append(location)
        del history[:-args.history_length]


def epinions_rows(args):
    frame = pd.read_csv(args.input, sep=args.separator)
    required = {"uid", "iid", "timestamp"}
    if required - set(frame):
        raise ValueError("Epinions interaction table requires uid, iid, timestamp column headers.")
    if "label" in frame:
        if not frame.label.isin([0, 1]).all():
            raise ValueError("Epinions labels must be binary.")
    elif "rating" in frame:
        frame["rating"] = pd.to_numeric(frame.rating, errors="raise")
        if frame.rating.isna().any():
            raise ValueError("Epinions ratings must not be missing.")
    else:
        raise ValueError("Epinions input requires a binary label or a numerical rating column.")
    if "title" not in frame:
        if not args.metadata:
            raise ValueError("Supply Epinions title columns or --metadata with iid,title columns.")
        metadata = pd.read_csv(args.metadata)
        frame = frame.merge(metadata[["iid", "title"]], on="iid", how="left", validate="many_to_one")
    if frame.title.isna().any():
        raise ValueError("Missing Epinions item metadata.")
    frame["timestamp"] = pd.to_numeric(frame.timestamp, errors="raise")
    histories = defaultdict(list)
    for row in frame.sort_values("timestamp", kind="stable").itertuples(index=False):
        history = histories[row.uid]
        yield {"history": history[-args.history_length:], "target": str(row.title)}
        positive = getattr(row, "label", None)
        if positive is None:
            positive = row.rating >= args.positive_threshold
        if positive:
            history.append(str(row.title))
            del history[:-args.history_length]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCES), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--history-length", type=int, default=10)
    parser.add_argument("--positive-threshold", type=float, default=5)
    parser.add_argument("--separator", default=",")
    args = parser.parse_args()
    if args.history_length < 1:
        parser.error("--history-length must be positive.")
    if args.source.startswith("amazon-") and not args.metadata:
        parser.error("Amazon sources require --metadata.")
    readers = {"epinions": epinions_rows, "lastfm": lastfm_rows, "mind": mind_rows,
               "amazon-sports": amazon_rows, "amazon-beauty": amazon_rows, "gowalla": gowalla_rows}
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as stream:
        for count, row in enumerate(readers[args.source](args), 1):
            prompt, target = render(row, SOURCES[args.source])
            stream.write(json.dumps({"source": args.source, "record_id": f"{args.source}:{count - 1}",
                                    "prompt": prompt, "target": target}, ensure_ascii=False, allow_nan=False) + "\n")
    if not count:
        raise ValueError("No external records produced.")
    inputs = [Path(args.input)] if Path(args.input).is_file() else sorted(Path(args.input).glob("*"))
    if args.metadata:
        inputs.append(Path(args.metadata))
    write_json(output.with_suffix(".manifest.json"), {"source": args.source, "records": count, "options": vars(args),
               "inputs": {str(p): sha256(p) for p in inputs if p.is_file()}, "output_sha256": sha256(output)})
    print(f"Serialized {count} records to {output}")


if __name__ == "__main__":
    main()
