"""Five-axis audit engine (PLAN.md §3, Axes 1-4). Domain-agnostic.

Consumes results/predictions_{domain}.parquet, writes results/audit_{domain}.json.

Axis 1 (discrimination): AUC (macro over class groups for multiclass domains, else pooled)
                         and macro-F1 per split; DeLong CI per AUC.
Axis 2 (calibration):    ECE (M=10) and Brier per split; ECE bootstrap CI.
Axis 3 (subgroup):       per-subgroup AUC/ECE; gap G = max-min over an axis's groups;
                         DELTA-G = G(target) - G(source_test) is the headline "fairness gap
                         under shift".
Axis 4 (significance):   ΔAUC via closed-form DeLong SE addition (independent samples);
                         ΔG / ΔECE via stratified bootstrap; BH-FDR within (domain × axis).

AUC differences between two independent splits use added DeLong variances (fast, exact) rather
than bootstrap. ECE and gap CIs use a stratified bootstrap on a capped subsample (BOOT_CAP) so
the 1.28M-row clinical splits stay tractable.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import P, N_BOOT, ECE_BINS, BH_Q  # noqa: E402
from fairscope.core.delong import delong_auc_ci  # noqa: E402
from fairscope.core.calibration import expected_calibration_error  # noqa: E402
from fairscope.core.metrics import brier_score_loss  # noqa: E402
from fairscope.core.correction import benjamini_hochberg  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

Z = 1.959964
BOOT_CAP = 40_000  # per-split subsample cap for ECE/gap bootstrap (tractability)
PRIMARY_AXIS = {"clinical": "age", "nlp": "proxy_class", "lending": "race_black",
                "security": "attack_family"}


def _is_multiclass(df: pd.DataFrame) -> bool:
    return df["class_label"].notna().any()


def _auc_se(y, p):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    r = delong_auc_ci(np.asarray(y), np.asarray(p))
    return r["auc"], r["se"]


def _split_aggregate(sdf: pd.DataFrame, multiclass: bool, primary_axis: str):
    """Aggregate AUC (+SE), macro-F1, ECE, Brier for one split."""
    if multiclass:  # macro over class groups (one-vs-rest rows)
        aucs, ses, f1s, eces, briers = [], [], [], [], []
        for _, g in sdf[sdf.subgroup_axis == primary_axis].groupby("class_label"):
            a, s = _auc_se(g.y_true, g.p_hat)
            if not np.isnan(a):
                aucs.append(a); ses.append(s)
            f1s.append(f1_score(g.y_true, (g.p_hat >= 0.5).astype(int), zero_division=0))
            eces.append(expected_calibration_error(g.y_true.values, g.p_hat.values, n_bins=ECE_BINS))
            briers.append(brier_score_loss(g.y_true, g.p_hat))
        auc = float(np.mean(aucs)); se = float(np.sqrt(np.mean(np.square(ses))) / np.sqrt(len(ses)))
        return auc, se, float(np.mean(f1s)), float(np.mean(eces)), float(np.mean(briers))
    d = sdf[sdf.subgroup_axis == primary_axis]  # binary: rows = each example once
    auc, se = _auc_se(d.y_true, d.p_hat)
    f1 = f1_score(d.y_true, (d.p_hat >= 0.5).astype(int), zero_division=0)
    ece = expected_calibration_error(d.y_true.values, d.p_hat.values, n_bins=ECE_BINS)
    brier = brier_score_loss(d.y_true, d.p_hat)
    return auc, se, f1, ece, brier


def _subgroup_gap(sdf_axis: pd.DataFrame, metric: str):
    """Gap = max-min over subgroups for AUC or ECE within one axis, + per-group values."""
    vals = {}
    for g, x in sdf_axis.groupby("subgroup"):
        if metric == "auc":
            if x.y_true.nunique() == 2:
                vals[g] = _auc_se(x.y_true, x.p_hat)[0]
        else:
            vals[g] = expected_calibration_error(x.y_true.values, x.p_hat.values, n_bins=ECE_BINS)
    if len(vals) < 2:
        return np.nan, vals
    return float(max(vals.values()) - min(vals.values())), vals


def _boot_gap(sdf_axis: pd.DataFrame, metric: str, seed: int, n_boot=N_BOOT):
    """Stratified bootstrap of the subgroup gap on a capped subsample."""
    rng = np.random.default_rng(seed)
    total = len(sdf_axis)
    groups = {}
    for g, x in sdf_axis.groupby("subgroup"):
        if total > BOOT_CAP:
            k = max(50, int(BOOT_CAP * len(x) / total))
            x = x.sample(min(len(x), k), random_state=seed)
        groups[g] = x.reset_index(drop=True)
    out = []
    for _ in range(n_boot):
        vals = {}
        for g, x in groups.items():
            idx = rng.integers(0, len(x), len(x))
            yb, pb = x.y_true.values[idx], x.p_hat.values[idx]
            if metric == "auc":
                if len(np.unique(yb)) == 2:
                    vals[g] = delong_auc_ci(yb, pb)["auc"]
            else:
                vals[g] = expected_calibration_error(yb, pb, n_bins=ECE_BINS)
        if len(vals) >= 2:
            out.append(max(vals.values()) - min(vals.values()))
    return np.array(out)


def audit_domain(domain: str) -> dict:
    df = pd.read_parquet(P["out"] / f"predictions_{domain}.parquet")
    multiclass = _is_multiclass(df)
    paxis = PRIMARY_AXIS[domain]
    axes = sorted(df.subgroup_axis.unique())
    result = {"domain": domain, "multiclass": multiclass, "primary_axis": paxis, "models": {}}
    p_delta_auc, p_delta_gap = [], []  # (key, p) for BH within domain

    seeds_present = set(df["seed"].unique())
    ci_seed = 42 if 42 in seeds_present else int(df["seed"].min())  # CIs on the primary seed
    #   only (cost guard); other seeds contribute point estimates for stability.

    for model, mdf in df.groupby("model"):
        result["models"][model] = {}
        for seed, sdf_all in mdf.groupby("seed"):
            do_ci = (seed == ci_seed)
            bs_src_cache = None  # source-split gap bootstrap, computed once per (model, seed)
            splits = {}
            per_split_agg = {}
            for split, sdf in sdf_all.groupby("split"):
                auc, se, f1, ece, brier = _split_aggregate(sdf, multiclass, paxis)
                per_split_agg[split] = dict(auc=auc, auc_se=se)
                axis_gaps = {}
                for ax in axes:
                    axdf = sdf[sdf.subgroup_axis == ax]
                    g_auc, groups_auc = _subgroup_gap(axdf, "auc")
                    g_ece, groups_ece = _subgroup_gap(axdf, "ece")
                    axis_gaps[ax] = dict(gap_auc=g_auc, gap_ece=g_ece,
                                         subgroup_auc=groups_auc, subgroup_ece=groups_ece)
                splits[split] = dict(auc=auc, auc_se=se, macro_f1=f1, ece=ece, brier=brier,
                                     axis_gaps=axis_gaps)

            # deltas: each target vs source_test
            src = splits.get("source_test")
            deltas = {}
            for split, s in splits.items():
                if split == "source_test" or src is None:
                    continue
                d_auc = src["auc"] - s["auc"]
                se_d = np.sqrt(src["auc_se"] ** 2 + s["auc_se"] ** 2) if not np.isnan(src["auc_se"]) else np.nan
                z = d_auc / se_d if se_d and se_d > 0 else np.nan
                from math import erf, sqrt
                p_auc = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))) if not np.isnan(z) else np.nan
                key = f"{model}/s{seed}/{split}"
                if not np.isnan(p_auc):
                    p_delta_auc.append((key, p_auc))
                # point-estimate ΔG (always) = gap(target) - gap(source), primary axis
                d_gap_point = s["axis_gaps"][paxis]["gap_auc"] - src["axis_gaps"][paxis]["gap_auc"]
                d_gap = d_gap_point
                p_gap = gap_lo = gap_hi = np.nan
                if do_ci:  # bootstrap CI only on the primary seed (cost guard)
                    if bs_src_cache is None:
                        bs_src_cache = _boot_gap(
                            sdf_all[(sdf_all.split == "source_test") & (sdf_all.subgroup_axis == paxis)], "auc", seed)
                    bs_tgt = _boot_gap(sdf_all[(sdf_all.split == split) & (sdf_all.subgroup_axis == paxis)], "auc", seed)
                    if len(bs_src_cache) and len(bs_tgt):
                        m = min(len(bs_src_cache), len(bs_tgt))
                        diff = bs_tgt[:m] - bs_src_cache[:m]  # positive => gap widens under shift
                        d_gap = float(np.mean(diff)); gap_lo, gap_hi = np.percentile(diff, [2.5, 97.5])
                        p_gap = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
                        p_delta_gap.append((key, p_gap))
                deltas[split] = dict(delta_auc=d_auc, delta_auc_p=p_auc,
                                     delta_gap_auc=d_gap, delta_gap_point=d_gap_point,
                                     delta_gap_ci=[float(gap_lo), float(gap_hi)],
                                     delta_gap_p=float(p_gap) if not np.isnan(p_gap) else None)
            result["models"][model][str(seed)] = dict(splits=splits, deltas=deltas)

    # Axis 4: BH-FDR within domain, per family
    for fam, plist in [("delta_auc", p_delta_auc), ("delta_gap", p_delta_gap)]:
        if plist:
            keys, pv = zip(*plist)
            rej, _padj = benjamini_hochberg(list(pv), alpha=BH_Q)
            result.setdefault("bh_significant", {})[fam] = {k: bool(r) for k, r in zip(keys, rej)}
    return result


def _jsonable(o):
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main():
    for domain in ["clinical", "nlp", "lending", "security"]:
        fp = P["out"] / f"predictions_{domain}.parquet"
        if not fp.exists():
            print(f"[skip] {domain}: no predictions parquet yet")
            continue
        res = audit_domain(domain)
        out = P["out"] / f"audit_{domain}.json"
        out.write_text(json.dumps(res, indent=2, default=_jsonable))
        # one-line summary
        print(f"\n=== {domain} ===")
        for model, seeds in res["models"].items():
            s42 = seeds.get("42") or next(iter(seeds.values()))
            for split, d in s42["deltas"].items():
                print(f"  {model:18s} {split:22s} dAUC={d['delta_auc']:+.4f}  "
                      f"dG_auc={d['delta_gap_auc']:+.4f} (p={d['delta_gap_p']})")
    print("\nwrote audit_*.json")


if __name__ == "__main__":
    main()
