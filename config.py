"""TrustShift configuration: ALL paths and constants live here (single PATHS block).

Every script imports from this module. Never hardcode a machine path elsewhere.
"""
from pathlib import Path

P = {
    "clinical_repo":     Path(r"D:\Projects\diabetes_prediction_project\federated"),
    "nlp_repo":          Path(r"D:\Projects\mental-health-fairness-nlp-main"),
    "hmda_features":     Path(r"D:\Projects\CATE-HMDA-Heterogeneous-Effects\data\features_panel.parquet"),
    "hmda_feature_sets": Path(r"D:\Projects\CATE-HMDA-Heterogeneous-Effects\data\feature_sets.json"),
    "ddos_notebook":     Path(r"C:\Users\Asus\Downloads\CrossDataset_DDoS_Colab.ipynb"),
    "fairscope":         Path(r"D:\Projects\fairscope"),
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

OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]

REQUIRED_COLUMNS = [
    "domain", "model", "seed", "split", "y_true", "class_label",
    "p_hat", "subgroup_axis", "subgroup", "row_id",
]
