"""TrustShift configuration: ALL paths and constants live here (single PATHS block).

Every script imports from this module. Never hardcode a machine path elsewhere.

The sibling-repo paths below default to this project's own development machine layout and are
only needed to regenerate predictions from raw source repos (most reproduction needs just the
committed results/ predictions). Override any of them with the matching TRUSTSHIFT_* environment
variable -- see README.md's Configuration section -- without editing this file.
"""
import os
from pathlib import Path


def _env_path(var: str, default: str) -> Path:
    return Path(os.environ.get(var, default))


P = {
    "clinical_repo":     _env_path("TRUSTSHIFT_CLINICAL_REPO", r"D:\Projects\diabetes_prediction_project\federated"),
    "nlp_repo":          _env_path("TRUSTSHIFT_NLP_REPO", r"D:\Projects\mental-health-fairness-nlp-main"),
    "hmda_features":     _env_path("TRUSTSHIFT_HMDA_FEATURES", r"D:\Projects\CATE-HMDA-Heterogeneous-Effects\data\features_panel.parquet"),
    "hmda_feature_sets": _env_path("TRUSTSHIFT_HMDA_FEATURE_SETS", r"D:\Projects\CATE-HMDA-Heterogeneous-Effects\data\feature_sets.json"),
    "ddos_notebook":     _env_path("TRUSTSHIFT_DDOS_NOTEBOOK", r"C:\Users\Asus\Downloads\CrossDataset_DDoS_Colab.ipynb"),
    "fairscope":         _env_path("TRUSTSHIFT_FAIRSCOPE", r"D:\Projects\fairscope"),
    "out":               Path(__file__).parent / "results",
    "data":              Path(__file__).parent / "data",
}

SEED = 42
SEEDS_NEW = [42, 7, 123]          # newly trained models (lending, security)
NLP_SEEDS = [42, 0, 1, 7, 123]    # saved multiseed predictions in the CPFE repo

N_BOOT = 2000        # bootstrap resamples for CIs (matches P4/P6 practice)
N_BOOT_ECE = 1000    # ECE bootstrap CIs (matches P4)
ECE_BINS = 10        # equal-width bins (P4 eq. 3)
BH_Q = 0.05          # Benjamini-Hochberg FDR level per (domain x axis) family
DC_CLIP = 10.0       # importance-weight clip for the concept-shift test

TARGET_CALIB_FRAC = 0.10          # held-out target calibration split (remediation L1)
SMALL_N = [100, 500, 1000]        # labeled target sizes for remediation L3

# Phase-1 IJDSA repair (PLAN_ijdsa.md T1.8, BLOCKER-5): source-only primary-model selection.
# audit/primary_model.py applies this MECHANICALLY -- highest source_test AUC, ties broken by
# lower source_test ECE, remaining ties broken by this fixed name-priority list. No target-split
# quantity may enter the decision. Every other model is still reported in full in T2_master.csv.
PRIMARY_MODEL_TIEBREAK_PRIORITY = {
    "clinical": ["fedavg", "xgb"],
    "nlp": ["bert", "roberta", "mentalbert", "mentalroberta"],
    "lending": ["lightgbm_temporal", "lightgbm_geo"],
    "security": ["lightgbm", "xgboost"],
}

OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]

REQUIRED_COLUMNS = [
    "domain", "model", "seed", "split", "y_true", "class_label",
    "p_hat", "subgroup_axis", "subgroup", "row_id",
]
