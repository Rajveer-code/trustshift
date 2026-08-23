"""Shift diagnosis (PLAN.md §3, Axis 5): decompose each shift into label / covariate / concept.

Two-stage protocol (PLAN_ijdsa.md T1.6):
  Stage A -- unlabeled screening: label prevalence Delta-pi (class metadata only, no target
             labels needed beyond what defines the split), domain-classifier AUC_dc (needs only
             unlabeled feature vectors from both sides).
  Stage B -- small-label confirmation: the concept-shift reweighting probe. This NEEDS target
             labels (auc_target) to compare against the reweighted source AUC, so it is NOT
             label-free and is never described as such (BLOCKER-4).

- label shift:    prevalence pi per split (+per class for nlp); Delta-pi.
                  clinical/nlp/lending: from the predictions parquet.
                  security: from the RAW, UNCAPPED harmonized population
                  (security_raw_prevalence.json), NOT the capped predictions parquet --
                  see PLAN_ijdsa.md BLOCKER-6. The capped parquet's prevalence is an artifact of
                  the per-family sampling cap applied for training-compute reasons on both sides
                  and is not the deployment-relevant quantity.
- covariate shift: domain-classifier AUC (source_test vs target on shared features, 5-fold CV,
                  out-of-fold) + top-10 Population Stability Index.
                  nlp: features = TF-IDF(SVD-50) of text + [length, type-token ratio].
                  lending: features = the model's numeric features (panel reload, distributional).
                  clinical: target-side BRFSS features are unrecoverable (PLAN.md §2a); covariate
                            shift is proxied by the KS distance between source/target p_hat.
                  security: NOT COMPUTED. No domain classifier has been fit for security features
                            (this is a genuine absence of evidence, not a null result -- see
                            probe_availability in the output and PLAN_ijdsa.md's probe-
                            availability matrix).
- concept shift:  importance-weighted source AUC vs unweighted vs target AUC, with out-of-fold
                  domain-classifier weights, a sampling-prior correction applied BEFORE clipping,
                  and effective-sample-size (ESS) reporting (PLAN_ijdsa.md T1.7). If ESS is too
                  small relative to n_source, the probe is marked non-adjudicable and MUST NOT be
                  used to assert or rule out concept shift -- computed for nlp only (text aligns
                  to source preds); for lending/clinical/security the feature<->prediction row
                  alignment is unavailable or the probe was not attempted, and this is stated
                  explicitly rather than silently defaulting to a taxonomy label.

Rule for the one-line label written into each JSON (`_summarize`):
  Every mechanism label requires a REAL, adjudicable probe result above threshold. If zero probes
  fire, or the only firing probe's own assumptions are directly contradicted by other measured
  evidence (e.g. a large Delta-pi where the positive-class attack families are measurably
  disjoint between source and target, contradicting label shift's P(X|Y)-stable assumption), the
  diagnosis is `inconclusive` -- a first-class outcome, not a failure of the framework
  (PLAN_ijdsa.md T1.6 / user instruction: "make inconclusive the true fallback").
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
ESS_FRAC_MIN = 0.10  # below this fraction of n_source, importance weighting is non-adjudicable


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
    """Balanced (equal n per side), out-of-fold domain-classifier AUC. Used only for the AUC_dc
    covariate-shift diagnostic; because Xs/Xt are capped to equal size here, the classifier's
    training prior is 0.5/0.5, so no sampling-prior correction is needed for this quantity
    (contrast with _importance_weights below, whose classifier is fit on natural, usually
    unequal, class sizes and therefore does need the correction)."""
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


def _importance_weights(Xs: np.ndarray, Xt: np.ndarray, seed=SEED, clip=DC_CLIP):
    """Out-of-fold, sampling-prior-corrected importance weights for reweighting Xs to look like
    Xt (PLAN_ijdsa.md T1.7). Returns (weights_for_Xs, diagnostics).

    The domain classifier here is fit on the NATURAL (possibly unequal) sizes of Xs and Xt, so
    its estimated P(D=target|x) reflects a training prior of n_t/(n_s+n_t), not 0.5. The correct
    density-ratio estimate is:
        p_target(x)/p_source(x) = [P(D=source)/P(D=target)] * [Phat(D=target|x)/Phat(D=source|x)]
                                 = (n_s/n_t) * p(x)/(1-p(x))
    The (n_s/n_t) correction is applied BEFORE clipping at `clip`, because clipping is a
    nonlinear operation on the weight scale -- correcting after clipping would clip the wrong
    points.
    """
    n_s, n_t = len(Xs), len(Xt)
    X = np.vstack([Xs, Xt]); y = np.r_[np.zeros(n_s), np.ones(n_t)]
    clf = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=32,
                         random_state=seed, n_jobs=-1, verbose=-1)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p_oof = cross_val_predict(clf, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    auc_dc_oof = float(roc_auc_score(y, p_oof))
    p_s = np.clip(p_oof[:n_s], 1e-3, 1 - 1e-3)
    prior_correction = n_s / n_t
    w_raw = prior_correction * p_s / (1 - p_s)
    w = np.clip(w_raw, 0, clip)
    ess = float((w.sum() ** 2) / (np.sum(w ** 2) + 1e-12))
    ess_frac = float(ess / n_s)
    diag = dict(auc_dc_oof=auc_dc_oof, ess=ess, ess_frac=ess_frac, n_source=int(n_s),
                n_target=int(n_t), prior_correction=float(prior_correction),
                max_weight=float(w.max()), median_weight=float(np.median(w)),
                p99_weight=float(np.percentile(w, 99)),
                adjudicable=bool(ess_frac >= ESS_FRAC_MIN))
    return w, diag


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


def _security_family_evidence() -> dict:
    """Whether the target's attack families were present in source TRAINING data (not merely
    in the source_test evaluation partition -- PLAN_ijdsa.md BLOCKER-6 follow-up). Read from
    security_label_space.csv, written by domains/security/run_pipeline.py."""
    fp = P["out"] / "security_label_space.csv"
    if not fp.exists():
        return {"available": False}
    tbl = pd.read_csv(fp)
    trained = set(tbl.loc[tbl.in_source_train, "family"]) - {"benign"}
    target_fams = set(tbl.loc[tbl.in_target, "family"]) - {"benign"}
    overlap = trained & target_fams
    return {"available": True, "trained_attack_families": sorted(trained),
            "target_attack_families": sorted(target_fams), "overlap": sorted(overlap),
            "disjoint": len(overlap) == 0}


# ---- domain diagnosis drivers ---------------------------------------------------
def diagnose(domain: str) -> dict:
    df = pd.read_parquet(P["out"] / f"predictions_{domain}.parquet")
    df42 = df[df.seed == 42] if 42 in set(df.seed.unique()) else df[df.seed == df.seed.min()]
    res = {"domain": domain, "label_shift": _label_shift(df42), "covariate": {}, "concept": {},
           "probe_availability": {"label_shift": True, "covariate": False, "concept": False}}

    if domain == "nlp":
        from audit.primary_model import select_primary_models
        concept_model = select_primary_models("nlp")[0]  # 2026-08-22 consistency fix: this was
        # hardcoded to "bert" regardless of which model the source-only rule selected as primary
        # (currently roberta) -- caught by the primary-model consistency audit
        # (results/primary_model_consistency_report.json). The concept probe now always uses
        # whichever model audit/primary_model.py selects, so the diagnosis and the main-table
        # numbers can never silently reference two different models.
        feats, texts = _nlp_text_features()
        src = feats["source_test"]
        for tgt in ["target_reddit", "target_twitter"]:
            res["covariate"][tgt] = {"auc_dc": _domain_classifier_auc(src, feats[tgt])}
        for tgt in ["target_reddit", "target_twitter"]:
            res["concept"][tgt] = _concept_nlp(df42, feats, tgt, model=concept_model)
        res["probe_availability"]["covariate"] = True
        res["probe_availability"]["concept"] = True
        res["concept_probe_model"] = concept_model

    elif domain == "lending":
        feats, names = _lending_features()
        src = feats["source_test"]
        for tgt in [k for k in feats if k != "source_test"]:
            auc_dc = _domain_classifier_auc(src, feats[tgt])
            psis = sorted(((n, _psi(src[:, i], feats[tgt][:, i])) for i, n in enumerate(names)),
                          key=lambda kv: -kv[1])[:10]
            res["covariate"][tgt] = {"auc_dc": auc_dc, "top_psi": dict(psis)}
        res["covariate"]["geo_note"] = ("geographic target (target_heldout_states) has no "
                                        "covariate probe -- AUC_dc computed for temporal years "
                                        "only. Phase-3 (T3.1) replaces this with a per-state "
                                        "severity ladder that will supersede this limitation.")
        res["concept"] = {"note": "feature<->prediction alignment unavailable; label+covariate reported"}
        res["probe_availability"]["covariate"] = True  # true for temporal targets only

    elif domain == "clinical":
        # target BRFSS features unrecoverable -> covariate proxied by p_hat KS distance. This is
        # a SCORE-distribution proxy, not a feature-space domain classifier, and is not treated
        # as equivalent evidence to AUC_dc (no "auc_dc" key is set).
        from audit.primary_model import select_primary_models
        ks_model = select_primary_models("clinical")[0]  # 2026-08-22 consistency fix: this
        # previously had NO model filter at all, silently pooling fedavg and xgb rows into one
        # KS-distance sample -- caught by the primary-model consistency audit.
        prim = df42[(df42.subgroup_axis == "age") & (df42.model == ks_model)]
        res["p_hat_ks_model"] = ks_model
        ps = prim[prim.split == "source_test"].p_hat.values
        for tgt in [s for s in prim.split.unique() if s != "source_test"]:
            pt = prim[prim.split == tgt].p_hat.values
            res["covariate"][tgt] = {"p_hat_ks": _ks(ps, pt),
                                     "note": "target-side features unrecoverable; score-distribution proxy, not AUC_dc"}
        res["concept"] = {"note": "target features unrecoverable; not computed"}

    elif domain == "security":
        raw = json.loads((P["out"] / "security_raw_prevalence.json").read_text())
        res["label_shift"] = {
            "source_test": {"pos": raw["source_cicddos2019"]["pos_rate"]},
            "target_cicids2017": {"pos": raw["target_cicids2017"]["pos_rate"]},
        }
        res["label_shift_note"] = (
            "Computed on the RAW (uncapped) harmonized, population-filtered data -- see "
            "security_raw_prevalence.json -- NOT the capped predictions parquet. The capped "
            f"parquet's prevalence is source={raw['source_cicddos2019']['pos_rate']:.3f}-ish "
            "artificially close to target because both sides are capped per-attack-family for "
            "training-compute reasons; the raw population prevalence is the deployment-relevant "
            "quantity and is substantially larger (BLOCKER-6 fix, PLAN_ijdsa.md)."
        )
        fam_ev = _security_family_evidence()
        res["family_disjoint"] = {"target_cicids2017": fam_ev.get("disjoint", False)}
        res["family_overlap_note"] = fam_ev
        res["covariate"] = {}
        res["concept"] = {}
        # both explicitly False: no domain classifier or reweighting probe has been fit on
        # security features. This is a genuine absence of evidence, not a null result.

    res["summary"] = _summarize(res)
    return res


def _ks(a, b):
    from scipy.stats import ks_2samp
    return float(ks_2samp(a, b).statistic)


def _concept_nlp(df42, feats, tgt, model: str) -> dict:
    d = df42[(df42.model == model) & (df42.class_label == "depression")]
    src_rows = d[d.split == "source_test"].sort_values("row_id")
    y_s, p_s = src_rows.y_true.values, src_rows.p_hat.values
    Xs = feats["source_test"][: len(y_s)]
    Xt = feats[tgt]
    w, diag = _importance_weights(Xs, Xt)
    auc_src = roc_auc_score(y_s, p_s)
    auc_src_rw = roc_auc_score(y_s, p_s, sample_weight=w)
    tgt_rows = df42[(df42.model == model) & (df42.class_label == "depression") & (df42.split == tgt)]
    auc_tgt = roc_auc_score(tgt_rows.y_true, tgt_rows.p_hat)
    out = {"auc_source": float(auc_src), "auc_source_reweighted": float(auc_src_rw),
           "auc_target": float(auc_tgt), "target": tgt, "model": model,
           "weighting_diagnostics": diag, "adjudicable": diag["adjudicable"]}
    if not diag["adjudicable"]:
        out["note"] = (
            f"effective sample size is {diag['ess_frac']:.1%} of n_source "
            f"(< {ESS_FRAC_MIN:.0%} threshold) -- the available source data do not provide "
            "sufficient overlap with the target for reliable importance-weighted adjudication. "
            "This does NOT mean covariate shift is ruled out; it means reweighting cannot "
            "adjudicate here."
        )
    return out


def _summarize(res: dict) -> dict:
    out = {}
    ls = res["label_shift"]
    src = ls.get("source_test", {})
    for tgt, tv in ls.items():
        if tgt == "source_test":
            continue
        dpi = max((abs(tv[k] - src.get(k, 0)) for k in tv), default=0.0)
        cov = res["covariate"].get(tgt, {})
        auc_dc = cov.get("auc_dc")  # only ever set by a real feature-space domain classifier
        con = res["concept"].get(tgt, {})

        evidence = []
        if isinstance(con, dict) and con.get("adjudicable"):
            total = con["auc_source"] - con["auc_target"]
            resid = con["auc_source_reweighted"] - con["auc_target"]
            if total > 0.02 and resid > 0.5 * total:
                evidence.append("concept")
        if dpi > 0.15:
            evidence.append("label")
        if auc_dc is not None and auc_dc > 0.65:
            evidence.append("covariate")

        family_disjoint = res.get("family_disjoint", {}).get(tgt, False)
        note = None
        if "label" in evidence and family_disjoint:
            diagnosis = "inconclusive"
            note = (f"large prevalence shift measured (|Delta-pi|={dpi:.3f}), but the positive-"
                     "class attack families are measurably disjoint between source training "
                     "data and target (see family_overlap_note) -- label shift's assumption "
                     "that P(X|Y) is stable is directly contradicted, so the shift cannot be "
                     "reduced to a prevalence change alone")
        elif not evidence:
            diagnosis = "inconclusive"
            note = "no probe produced adjudicable evidence above threshold for this target"
        elif len(evidence) == 1:
            diagnosis = evidence[0]
        else:
            diagnosis = "mixed"
            note = f"multiple probes fired: {evidence}"

        entry = {"max_delta_pi": float(dpi), "auc_dc": auc_dc, "diagnosis": diagnosis}
        if note:
            entry["note"] = note
        out[tgt] = entry
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
            note = f"  [{sm['note']}]" if "note" in sm else ""
            print(f"  {tgt:22s} dpi={sm['max_delta_pi']:.3f} auc_dc={sm['auc_dc']} -> {sm['diagnosis']}{note}")
    print("\nwrote diagnosis_*.json")


if __name__ == "__main__":
    main()
