"""NLP domain adapter: Kaggle mental-health corpus (source) -> Reddit / Twitter (target).

Reuses saved per-seed predictions from the CPFE repo (PLAN.md §2b) — NO retraining.
Source CSVs: {nlp_repo}/outputs/results/multiseed/{model}_{platform}_seed{s}_predictions.csv
Columns used: label (0=normal,1=depression,2=anxiety,3=stress), prob_{class}, platform.

Model set on disk (bert, roberta, mentalbert, mentalroberta) differs from the P4 public
manuscript set; TrustShift reports the models actually saved. These four are not
GoEmotions-pretrained, so the P4 GoEmotions-Reddit non-independence caveat does not apply.

Design (PLAN.md §3): one-vs-rest per proxy class. subgroup_axis='proxy_class', subgroup=class.
Per-split macro-AUC = mean over the four class groups; the class spread is the subgroup gap.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import P, NLP_SEEDS  # noqa: E402
from schema import save_validated  # noqa: E402

MULTISEED = P["nlp_repo"] / "outputs" / "results" / "multiseed"
MODELS = ["bert", "roberta", "mentalbert", "mentalroberta"]
CLASSES = {0: "normal", 1: "depression", 2: "anxiety", 3: "stress"}
PLATFORM_SPLIT = {"kaggle": "source_test", "reddit": "target_reddit", "twitter": "target_twitter"}


def build() -> pd.DataFrame:
    frames = []
    for model in MODELS:
        for platform, split in PLATFORM_SPLIT.items():
            for seed in NLP_SEEDS:
                fp = MULTISEED / f"{model}_{platform}_seed{seed}_predictions.csv"
                if not fp.exists():
                    print(f"  [skip missing] {fp.name}")
                    continue
                df = pd.read_csv(fp)
                for c_idx, c_name in CLASSES.items():
                    y = (df["label"].values == c_idx).astype(int)
                    p = df[f"prob_{c_name}"].values.astype(float)
                    n = len(df)
                    frames.append(pd.DataFrame({
                        "domain": "nlp", "model": model, "seed": seed, "split": split,
                        "y_true": y, "class_label": c_name,
                        "p_hat": np.clip(p, 0.0, 1.0),
                        "subgroup_axis": "proxy_class", "subgroup": c_name,
                        "row_id": [f"nlp_{model}_{platform}_s{seed}_{c_name}_{i}" for i in range(n)],
                    }))
    return pd.concat(frames, ignore_index=True)


def main():
    df = build()
    out = P["out"] / "predictions_nlp.parquet"
    save_validated(df, out)

    from sklearn.metrics import roc_auc_score
    print("\n-- sanity macro-AUC per platform (seed 42, mean over classes) --")
    for model in MODELS:
        line = [f"{model:14s}"]
        for split in ["source_test", "target_reddit", "target_twitter"]:
            sub = df[(df.model == model) & (df.seed == 42) & (df.split == split)]
            aucs = [roc_auc_score(g.y_true, g.p_hat) for _, g in sub.groupby("class_label")
                    if g.y_true.nunique() == 2]
            line.append(f"{split.split('_')[-1]}={np.mean(aucs):.3f}")
        print("  " + "  ".join(line))
    print("  anchor (P4): kaggle ~0.983-0.987; reddit drop ~30-35%; twitter ~38-40%")


if __name__ == "__main__":
    main()
