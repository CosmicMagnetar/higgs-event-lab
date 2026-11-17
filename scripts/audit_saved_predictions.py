"""Recompute exported test metrics and verify row/label alignment without training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def audit(root: Path) -> dict:
    rows = []
    reference = None
    for path in sorted(root.glob("*/test_predictions.csv")):
        metadata = json.loads((path.parent / "metrics.json").read_text())
        frame = pd.read_csv(path).sort_values("row_index")
        if frame.row_index.duplicated().any():
            raise ValueError(f"Duplicate row IDs: {path.parent.name}")
        identity = frame[["row_index", "class_label"]].reset_index(drop=True)
        if reference is None:
            reference = identity
        if not identity.equals(reference):
            raise ValueError(f"Different test rows or labels: {path.parent.name}")
        y, p = frame.class_label, frame.score
        measured = dict(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                        brier_score=brier_score_loss(y, p))
        saved = metadata["evaluation"]["test"]
        differences = {key: abs(value - saved[key]) for key, value in measured.items()}
        rows.append(dict(run=path.parent.name, rows=len(frame), recomputed=measured,
                         maximum_absolute_difference=max(differences.values()),
                         matches=bool(all(np.isclose(measured[k], saved[k], atol=1e-7, rtol=0) for k in measured))))
    return dict(scope="Saved test predictions only; not an independent training reproduction",
                shared_rows_and_labels=bool(rows), runs=rows,
                passed=bool(rows) and all(row["matches"] for row in rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("reports/saved_prediction_audit.json"))
    args = parser.parse_args()
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(result['runs'])} prediction files checked; passed={result['passed']}")
    raise SystemExit(0 if result["passed"] else 1)
