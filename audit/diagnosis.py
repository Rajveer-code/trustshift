"""Shift diagnosis (PLAN.md §3, Axis 5): decompose each shift into label / covariate / concept.

- label shift:    prevalence pi per split (+per class for nlp); Delta-pi.  [all domains, from parquet]
- covariate shift: domain-classifier AUC (source_test vs target on shared features, 5-fold CV)
                   + top-10 Population Stability Index.
                   nlp: features = TF-IDF(SVD-50) of text + [length, type-token ratio].
                   lending: features = the model's numeric features (panel reload, distributional).
                   clinical: target-side BRFSS features are unrecoverable (PLAN.md §2a); covariate
                             shift is proxied by the KS distance between source/target p_hat.
- concept shift:  importance-weighted source AUC vs unweighted vs target AUC. Requires features
                  aligned to the scored rows -> computed for nlp (text aligns to source preds);
                  for lending/clinical the feature<->prediction row alignment is unavailable, so
                  concept is left uncomputed and flagged (label + covariate already characterize
                  most of the shift there).

Rule for the one-line label written into each JSON:
  label     if |max Delta-pi| > 0.15 and dominates;
  concept   if (source_reweighted_auc - target_auc) > 0.5 * (source_auc - target_auc);
  covariate if AUC_dc > 0.65 and not concept;
  else mixed.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import P, SEED, DC_CLIP  # noqa: E402
from audit.engine import _jsonable  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.model_selection import cross_val_predict, StratifiedKFold  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402

DC_CAP = 20_000  # per-side cap for the domain classifier


def _label_shift(df: pd.DataFrame) -> dict:
    out = {}
    if df["class_label"].notna().any():
        for split, s in df.groupby("split"):
            out[split] = {c: float((g.y_true == 1).mean()) for c, g in s.groupby("class_label")}
    else:
        prim = df[df.subgroup_axis == df.subgroup_axis.iloc[0]]
        for split, s in prim.groupby("split"):
            out[split] = {"pos": float((s.y_true == 1).mean())}
    return out


def _psi(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
    qs = np.quantile(a, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    pa = np.histogram(a, qs)[0] / len(a) + 1e-6
    pb = np.histogram(b, qs)[0] / len(b) + 1e-6
    return float(np.sum((pa - pb) * np.log(pa / pb)))


def _domain_classifier_auc(Xs: np.ndarray, Xt: np.ndarray, seed=SEED) -> float:
    rng = np.random.default_rng(seed)
    Xs = Xs[rng.choice(len(Xs), min(len(Xs), DC_CAP), replace=False)]
    Xt = Xt[rng.choice(len(Xt), min(len(Xt), DC_CAP), replace=False)]
    X = np.vstack([Xs, Xt])
    y = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    clf = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=32,
                         random_state=seed, n_jobs=-1, verbose=-1)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    return float(roc_auc_score(y, p))


# ---- per-domain feature builders ------------------------------------------------
def _nlp_text_features():
    """seed-42 text per platform -> shared TF-IDF(SVD-50) + [len, ttr]. Returns dict split->X."""
    ms = P["nlp_repo"] / "outputs" / "results" / "multiseed"
    texts, splits = {}, {"kaggle": "source_test", "reddit": "target_reddit", "twitter": "target_twitter"}
    for plat in splits:
        df = pd.read_csv(ms / f"bert_{plat}_seed42_predictions.csv")
        texts[plat] = df["text"].fillna("").astype(str).tolist()
    all_text = [t for plat in texts for t in texts[plat]]
    vec = TfidfVectorizer(max_features=5000, min_df=5).fit(all_text)
    svd = TruncatedSVD(50, random_state=SEED).fit(vec.transform(all_text))

    def feats(tlist):
        sv = svd.transform(vec.transform(tlist))
        length = np.array([[len(t.split())] for t in tlist], dtype=float)
        ttr = np.array([[len(set(t.split())) / max(1, len(t.split()))] for t in tlist])
        return np.hstack([sv, length, ttr])

    return {splits[p]: feats(texts[p]) for p in texts}, {splits[p]: texts[p] for p in texts}


def _lending_features():
    """Fresh distributional samples of model features per split (alignment-free)."""
    from domains.lending.train import FEATURES, _load_panel
    panel = _load_panel()
    out = {}
    src = panel[panel.year.isin([2020, 2021])].sample(min(DC_CAP, len(panel)), random_state=SEED)
    out["source_test"] = src[FEATURES].to_numpy()
    for yr in (2022, 2023, 2024):
        s = panel[panel.year == yr]
        out[f"target_{yr}"] = s.sample(min(DC_CAP, len(s)), random_state=SEED)[FEATURES].to_numpy()
    del panel
    return out, FEATURES


# ---- domain diagnosis drivers ---------------------------------------------------
def diagnose(domain: str) -> dict:
    df = pd.read_parquet(P["out"] / f"predictions_{domain}.parquet")
    df42 = df[df.seed == 42] if 42 in set(df.seed.unique()) else df[df.seed == df.seed.min()]
    res = {"domain": domain, "label_shift": _label_shift(df42), "covariate": {}, "concept": {}}

    if domain == "nlp":
        feats, texts = _nlp_text_features()
        src = feats["source_test"]
        # source predictions for the concept reweight (primary axis = macro; use depression class)
        for tgt in ["target_reddit", "target_twitter"]:
            res["covariate"][tgt] = {"auc_dc": _domain_classifier_auc(src, feats[tgt])}
        # concept: reweight source (bert, depression OvR) by domain-classifier odds
        for tgt in ["target_reddit", "target_twitter"]:
            res["concept"][tgt] = _concept_nlp(df42, feats, tgt)

    elif domain == "lending":
        feats, names = _lending_features()
        src = feats["source_test"]
        for tgt in [k for k in feats if k != "source_test"]:
            auc_dc = _domain_classifier_auc(src, feats[tgt])
            psis = sorted(((n, _psi(src[:, i], feats[tgt][:, i])) for i, n in enumerate(names)),
                          key=lambda kv: -kv[1])[:10]
            res["covariate"][tgt] = {"auc_dc": auc_dc, "top_psi": dict(psis)}
        res["covariate"]["geo_note"] = "geographic target uses the same feature space; AUC_dc for temporal years shown"
        res["concept"] = {"note": "feature<->prediction alignment unavailable; label+covariate reported"}

    elif domain == "clinical":
        # target BRFSS features unrecoverable -> covariate proxied by p_hat KS distance
        prim = df42[df42.subgroup_axis == "age"]
        ps = prim[prim.split == "source_test"].p_hat.values
        for tgt in [s for s in prim.split.unique() if s != "source_test"]:
            pt = prim[prim.split == tgt].p_hat.values
            res["covariate"][tgt] = {"p_hat_ks": _ks(ps, pt),
                                     "note": "target-side features unrecoverable; score-distribution proxy"}
        res["concept"] = {"note": "target features unrecoverable; not computed"}

    res["summary"] = _summarize(res)
    return res


def _ks(a, b):
    from scipy.stats import ks_2samp
    return float(ks_2samp(a, b).statistic)


def _concept_nlp(df42, feats, tgt) -> dict:
    d = df42[(df42.model == "bert") & (df42.class_label == "depression")]
    src_rows = d[d.split == "source_test"].sort_values("row_id")
    y_s, p_s = src_rows.y_true.values, src_rows.p_hat.values
    Xs = feats["source_test"][: len(y_s)]
    Xt = feats[tgt]
    # domain classifier for weights on source
    rng = np.random.default_rng(SEED)
    X = np.vstack([Xs, Xt]); y = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    clf = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=32,
                         random_state=SEED, n_jobs=-1, verbose=-1).fit(X, y)
    pt = np.clip(clf.predict_proba(Xs)[:, 1], 1e-3, 1 - 1e-3)
    w = np.clip(pt / (1 - pt), 0, DC_CLIP)
    auc_src = roc_auc_score(y_s, p_s)
    auc_src_rw = roc_auc_score(y_s, p_s, sample_weight=w)
    tgt_rows = df42[(df42.model == "bert") & (df42.class_label == "depression") & (df42.split == tgt)]
    auc_tgt = roc_auc_score(tgt_rows.y_true, tgt_rows.p_hat)
    return {"auc_source": float(auc_src), "auc_source_reweighted": float(auc_src_rw),
            "auc_target": float(auc_tgt)}


def _summarize(res: dict) -> dict:
    out = {}
    ls = res["label_shift"]
    src = ls.get("source_test", {})
    for tgt, tv in ls.items():
        if tgt == "source_test":
            continue
        dpi = max((abs(tv[k] - src.get(k, 0)) for k in tv), default=0.0)
        cov = res["covariate"].get(tgt, {})
        auc_dc = cov.get("auc_dc")
        con = res["concept"].get(tgt, {})
        label = "mixed"
        if isinstance(con, dict) and "auc_source" in con:
            total = con["auc_source"] - con["auc_target"]
            resid = con["auc_source_reweighted"] - con["auc_target"]
            if total > 0.02 and resid > 0.5 * total:
                label = "concept"
        if label == "mixed" and dpi > 0.15:
            label = "label"
        if label == "mixed" and auc_dc and auc_dc > 0.65:
            label = "covariate"
        out[tgt] = {"max_delta_pi": float(dpi), "auc_dc": auc_dc, "diagnosis": label}
    return out


def main():
    for domain in ["clinical", "nlp", "lending", "security"]:
        if not (P["out"] / f"predictions_{domain}.parquet").exists():
            print(f"[skip] {domain}")
            continue
        res = diagnose(domain)
        (P["out"] / f"diagnosis_{domain}.json").write_text(json.dumps(res, indent=2, default=_jsonable))
        print(f"\n=== {domain} diagnosis ===")
        for tgt, sm in res["summary"].items():
            print(f"  {tgt:22s} dpi={sm['max_delta_pi']:.3f} auc_dc={sm['auc_dc']} -> {sm['diagnosis']}")
    print("\nwrote diagnosis_*.json")


if __name__ == "__main__":
    main()
