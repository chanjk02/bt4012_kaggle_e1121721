"""V19: 65/35 rank blend of V12 and a recent-14-step CatBoost model (1000 trees).
Run: python train_v19.py --data /path/to/competition-data --out ./BT4012_V19_run
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
import train_v12

FEATURES = train_v12.FEATURES
CATEGORICAL = ["feat_3", "feat_112", "feat_113", "feat_148", "feat_149"]
HORIZON = 14
ITERATIONS = 1000

def as_catboost_frame(frame):
    x = frame[FEATURES].copy()
    for col in CATEGORICAL:
        x[col] = x[col].round(8).astype(str)
    return x

def build_catboost():
    return CatBoostClassifier(
        iterations=ITERATIONS, depth=6, learning_rate=0.035,
        loss_function="Logloss", eval_metric="AUC", l2_leaf_reg=6,
        random_seed=4012, verbose=False, thread_count=4,
        allow_writing_files=False,
    )

def run(data_path, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tr, te, sample = train_v12.load_data(data_path)
    known = tr.label.notna().to_numpy()
    y = tr.loc[known, "label"].astype(int).to_numpy()
    steps = tr.loc[known, "time_step"].to_numpy()
    recent = steps > int(steps.max()) - HORIZON
    model_dir = out / "models"; model_dir.mkdir(exist_ok=True)
    v12_dir = model_dir / "v12"; v12_dir.mkdir(exist_ok=True)
    _, p12, diag = train_v12.predict(
        tr.loc[known, FEATURES], y, te[FEATURES], steps, v12_dir
    )
    if int(recent.sum()) < 1000:
        raise ValueError("Too few known labels in the recent 14-step window")
    model = build_catboost()
    model.fit(as_catboost_frame(tr.loc[known, FEATURES].loc[recent]),
              y[recent], cat_features=CATEGORICAL)
    p_cat = model.predict_proba(as_catboost_frame(te[FEATURES]))[:, 1]
    model.save_model(str(model_dir / "catboost_recent14_1000.cbm"))
    rank = lambda p: rankdata(p, method="average") / len(p)
    p19 = 0.65 * rank(p12) + 0.35 * rank(p_cat)
    def submission(pred):
        frame = sample[["index"]].merge(
            pd.DataFrame({"index": te["index"], "target": pred}),
            on="index", how="left", sort=False, validate="one_to_one")
        assert len(frame) == len(te)
        assert np.array_equal(frame["index"].to_numpy(), sample["index"].to_numpy())
        assert frame.target.notna().all() and frame.target.between(0, 1).all()
        return frame
    outputs = {
        "submission_v12.csv": submission(p12),
        "submission_v19.csv": submission(p19),
    }
    for name, frame in outputs.items():
        train_v12.write_csv(frame, out / name)
    meta = {
        "version": "V19",
        "recipe": "65% rank(V12) + 35% rank(recent-14 categorical CatBoost, 1000 trees)",
        "training_rows": int(known.sum()),
        "unknown_labels_excluded": int((~known).sum()),
        "recent14_labeled_rows": int(recent.sum()),
        "prediction_rows": len(te),
        "categorical_features": CATEGORICAL,
        "catboost_iterations": ITERATIONS,
        "catboost_depth": 6,
        "catboost_l2_leaf_reg": 6,
        "catboost_random_seed": 4012,
        "public_scores": {"V16_fallback": 0.94832, "V18": 0.94801, "V19": 0.94845},
        "validation_screen": "Across the four reused chronological cutoffs, 1000 trees exceeded 750 trees on each; mean blended AUC increased by about 0.000030. This is weak screening evidence, not a public-score guarantee.",
        "future_labels_used": False,
        "unknown_labels_pseudo_labeled": False,
        "v12_diagnostics": diag,
        "catboost_version": __import__("catboost").__version__,
    }
    for name in outputs:
        meta[name + "_sha256"] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    (out / "run_metadata_v19.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return outputs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="BT4012_V19_run")
    args = ap.parse_args()
    run(args.data or train_v12.find_data(), args.out)
