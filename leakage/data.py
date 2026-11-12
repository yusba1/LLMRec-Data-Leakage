"""Shared input contract for corpus construction and downstream recommendation."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "valid", "test")
REQUIRED = {"uid", "iid", "title", "his", "his_title", "label"}


def as_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
        if not isinstance(parsed, (list, tuple)):
            raise ValueError("A history/tag field must contain a list.")
        return list(parsed)
    raise ValueError(f"Expected a list, got {type(value).__name__}.")


def validate_frame(frame, name):
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{name}: empty split")
    frame = frame.copy().reset_index(drop=True)
    for column in ("uid", "iid", "label"):
        values = pd.to_numeric(frame[column], errors="raise")
        if values.isna().any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{name}: {column} must contain integers.")
        frame[column] = values.astype("int64")
    if (frame[["uid", "iid"]] <= 0).any().any():
        raise ValueError(f"{name}: user and target-item IDs must start at 1; ID 0 is reserved for padding.")
    if not frame.label.isin([0, 1]).all():
        raise ValueError(f"{name}: label must be binary (0 or 1).")
    if frame.title.isna().any() or frame.title.astype(str).str.strip().eq("").any():
        raise ValueError(f"{name}: missing item titles")
    frame["title"] = frame.title.astype(str)
    frame["his"] = frame.his.map(as_list)
    frame["his_title"] = frame.his_title.map(as_list)
    for index, (ids, titles) in enumerate(zip(frame.his, frame.his_title)):
        if len(ids) != len(titles):
            raise ValueError(f"{name}, row {index}: history IDs/titles have different lengths.")
        if any(int(item) != item or int(item) < 0 for item in ids):
            raise ValueError(f"{name}, row {index}: invalid history item IDs")
    if "record_id" not in frame:
        frame["record_id"] = [f"{name}:{i}" for i in range(len(frame))]
    frame["record_id"] = frame.record_id.astype(str)
    if frame.record_id.duplicated().any():
        raise ValueError(f"{name}: duplicate record_id")
    return frame


def load_splits(directory):
    directory = Path(directory)
    frames = {split: validate_frame(pd.read_pickle(directory / f"{split}_ood2.pkl"), split)
              for split in SPLITS}
    identifiers = pd.concat([f.record_id for f in frames.values()], ignore_index=True)
    if identifiers.duplicated().any():
        raise ValueError("record_id must be unique across train, valid, and test.")
    return frames


def dimensions(frames):
    users = max(int(f.uid.max()) for f in frames.values()) + 1
    items = max(int(f.iid.max()) for f in frames.values()) + 1
    for frame in frames.values():
        for history in frame.his:
            if history:
                items = max(items, max(int(i) for i in history) + 1)
    return users, items


def serialize_history(titles):
    titles = [str(t).strip() for t in titles if str(t).strip()]
    return "User interaction history: " + ", ".join(titles) + ". Predict the next target item that aligns with the user's preference.\nTarget:"


def recommendation_samples(frame, history_length=10):
    """Expose only input features and labels; record IDs remain in the frame."""
    for row in frame.itertuples(index=False):
        pairs = [(int(i), str(t)) for i, t in zip(row.his, row.his_title) if int(i) != 0]
        pairs = pairs[-history_length:]
        ids = [i for i, _ in pairs]
        padded = [0] * (history_length - len(ids)) + ids
        yield {
            "UserID": int(row.uid), "TargetItemID": int(row.iid),
            "TargetItemTitle": '"' + row.title.strip() + '"',
            "InteractedItemTitles": ", ".join('"' + t + '"' for _, t in pairs),
            "InteractedItemIDs_pad": np.array(padded, dtype=np.int64),
            "InteractedNum": len(pairs), "sas_seq": np.array(padded, dtype=np.int64),
            "label": int(row.label),
        }


class RecommendationDataset:
    def __init__(self, frame, history_length=10):
        self.frame = frame
        self.history_length = history_length

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        return next(recommendation_samples(self.frame.iloc[index:index + 1], self.history_length))
