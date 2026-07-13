"""Security domain: CIC-DDoS2019 (source) -> CICIDS2017 (target) intrusion detection.

Ported from CrossDataset_DDoS_Colab.ipynb (PLAN.md §2d) with SMOKE_TEST removed — real run.
Data: cleaned parquet mirrors dhoogla/cicddos2019 + dhoogla/cicids2017 (Kaggle, via kagglehub).
Harmonization (leak-safe) is copied verbatim in spirit from the notebook: strip headers, drop
identity/socket columns, resolve duplicate Fwd Header Length, coerce inf->NaN, intersect the
numeric schema, clip external features to the training range.

Subgroup axis (PLAN.md §3, operational — NOT demographic): attack_family. Because families are
all-positive, an AUC-gap is undefined; the engine therefore reads security's aggregate AUC/ECE
shift (each example once, subgroup=family|benign), and this script separately writes per-family
detection RECALL (source vs target) to results/security_family_recall.json — the honest
operational-reliability metric.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import P, SEEDS_NEW, SEED  # noqa: E402
from schema import save_validated  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

SAMPLE_N_PER_CLASS = 20_000
LEAKY_COLS = {"unnamed: 0", "flow id", "fwd id", "source ip", "src ip", "destination ip",
              "dst ip", "source port", "src port", "destination port", "dst port",
              "timestamp", "simillarhttp", "similarhttp", "inbound"}
DDOS_MARKERS = ("ddos", "dos", "syn", "udp", "ldap", "mssql", "netbios", "snmp", "ssdp",
                "ntp", "dns", "tftp", "portmap", "flood", "hulk", "goldeneye", "slowloris",
                "slowhttptest", "loit")


def _load_parquet_dir(path: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(path) / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {path}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _find_label_col(df):
    for c in ("Label", "label", "Class", "class"):
        if c in df.columns:
            return c
    raise KeyError("no label column")


def harmonize(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.drop(columns=[c for c in df.columns if c.endswith(".1")], errors="ignore")
    label_col = _find_label_col(df)
    df = df.drop(columns=[c for c in df.columns if c.lower() in LEAKY_COLS], errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df, label_col


def to_binary(labels):
    low = labels.astype(str).str.lower()
    return low.apply(lambda s: s not in ("benign", "normal") and any(m in s for m in DDOS_MARKERS)
                     ).to_numpy().astype(int)


def _family(labels):
    """Coarse attack-family label from the raw multiclass name (benign -> 'benign')."""
    low = labels.astype(str).str.lower()
    def fam(s):
        if s in ("benign", "normal"):
            return "benign"
        for k in ("syn", "udp", "ldap", "mssql", "netbios", "snmp", "ssdp", "ntp", "dns",
                  "tftp", "portmap", "hulk", "goldeneye", "slowloris", "slowhttp", "ddos"):
            if k in s:
                return k
        return "other_attack"
    return low.map(fam)


def _cap(df, by, n, seed):
    parts = [df.loc[idx].sample(min(len(idx), n), random_state=seed)
             for idx in df.groupby(by).groups.values()]
    return pd.concat(parts, ignore_index=True)


def build() -> pd.DataFrame:
    import truststore  # use the Windows cert store (a proxy/AV injects a root CA not in certifi)
    truststore.inject_into_ssl()
    import kagglehub
    p19 = kagglehub.dataset_download("dhoogla/cicddos2019")
    p17 = kagglehub.dataset_download("dhoogla/cicids2017")
    print("downloaded:", p19, "|", p17)
    df19, lab19 = harmonize(_load_parquet_dir(p19))
    df17, lab17 = harmonize(_load_parquet_dir(p17))

    num19 = df19.drop(columns=[lab19]).select_dtypes("number").columns
    num17 = df17.drop(columns=[lab17]).select_dtypes("number").columns
    FEATURES = sorted(set(num19) & set(num17))
    assert FEATURES, "no shared features"
    print(f"shared features: {len(FEATURES)}")

    df19["_ybin"] = to_binary(df19[lab19]); df19["_fam"] = _family(df19[lab19])
    df17["_ybin"] = to_binary(df17[lab17]); df17["_fam"] = _family(df17[lab17])
    low17 = df17[lab17].astype(str).str.lower()
    df17 = df17[(low17.isin(["benign", "normal"])) | (df17["_ybin"] == 1)].reset_index(drop=True)

    df19 = _cap(df19, "_fam", SAMPLE_N_PER_CLASS, SEED)
    df17 = _cap(df17, "_ybin", SAMPLE_N_PER_CLASS, SEED)

    X19 = df19[FEATURES].fillna(df19[FEATURES].median()).to_numpy(np.float32)
    X17 = df17[FEATURES].fillna(df17[FEATURES].median()).to_numpy(np.float32)
    y19, y17 = df19["_ybin"].to_numpy(), df17["_ybin"].to_numpy()
    fam19, fam17 = df19["_fam"].to_numpy(), df17["_fam"].to_numpy()

    Xtr, Xte, ytr, yte, ftr, fte = train_test_split(
        X19, y19, fam19, test_size=0.2, random_state=SEED, stratify=y19)
    scaler = MinMaxScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    X17_s = np.clip(scaler.transform(X17), 0, 1)

    frames, recall = [], {}
    for seed in SEEDS_NEW:
        sw = compute_sample_weight("balanced", ytr)
        models = {
            "lightgbm": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=64,
                                       subsample=0.8, colsample_bytree=0.8, random_state=seed,
                                       n_jobs=-1, verbose=-1),
            "xgboost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=8,
                                     subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                     eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0),
        }
        for name, m in models.items():
            m.fit(Xtr_s, ytr, sample_weight=sw)
            for split, Xs, ys, fs in [("source_test", Xte_s, yte, fte),
                                      ("target_cicids2017", X17_s, y17, fam17)]:
                p = m.predict_proba(Xs)[:, 1]
                n = len(ys)
                frames.append(pd.DataFrame({
                    "domain": "security", "model": name, "seed": seed, "split": split,
                    "y_true": ys.astype(int), "class_label": None, "p_hat": np.clip(p, 0, 1),
                    "subgroup_axis": "attack_family",
                    "subgroup": np.where(ys == 0, "benign", fs),
                    "row_id": [f"sec_{name}_{split}_s{seed}_{i}" for i in range(n)]}))
                if name == "lightgbm" and seed == SEED:  # per-family recall at 0.5 threshold
                    pred = (p >= 0.5).astype(int)
                    fam_rc = {}
                    for fam in np.unique(fs[ys == 1]):
                        mask = (fs == fam) & (ys == 1)
                        fam_rc[str(fam)] = float(pred[mask].mean())
                    recall[split] = fam_rc
    (P["out"] / "security_family_recall.json").write_text(json.dumps(recall, indent=2))
    return pd.concat(frames, ignore_index=True)


def main():
    df = build()
    save_validated(df, P["out"] / "predictions_security.parquet")
    print("\n-- security detector AUC (lightgbm seed42) --")
    d = df[(df.model == "lightgbm") & (df.seed == SEED)]
    for split in ["source_test", "target_cicids2017"]:
        s = d[d.split == split]
        print(f"  {split:20s} AUC={roc_auc_score(s.y_true, s.p_hat):.4f}  n={len(s):,}")
    rc = json.loads((P["out"] / "security_family_recall.json").read_text())
    print("  per-family recall (source vs target):")
    fams = sorted(set(rc.get('source_test', {})) | set(rc.get('target_cicids2017', {})))
    for f in fams:
        print(f"    {f:14s} src={rc.get('source_test',{}).get(f,'-')} tgt={rc.get('target_cicids2017',{}).get(f,'-')}")


if __name__ == "__main__":
    main()
