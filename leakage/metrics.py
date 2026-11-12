"""Recommendation metrics on continuous scores, with equal weight per eligible user."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def ranking_metrics(users, labels, scores):
    users, labels, scores = np.asarray(users), np.asarray(labels), np.asarray(scores)
    if not (users.ndim == labels.ndim == scores.ndim == 1 and len(users) == len(labels) == len(scores)):
        raise ValueError("users, labels, and scores must be equal-length vectors.")
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Nonfinite scores or nonbinary labels.")
    auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None
    frame = pd.DataFrame({"uid": users, "label": labels, "score": scores})
    values = [float(roc_auc_score(g.label, g.score)) for _, g in frame.groupby("uid", sort=False)
              if g.label.nunique() == 2]
    return {"AUC": auc, "UAUC": float(np.mean(values)) if values else None,
            "n_records": len(labels), "n_users": int(frame.uid.nunique()), "eligible_users": len(values)}
