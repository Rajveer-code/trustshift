"""One-command reproduction of every result downstream of the prediction files.

    python run_all.py

Regenerates audit -> diagnosis -> remediation -> meta-analysis -> tables -> figures from the
committed results/predictions_*.parquet, then checks the regenerated master table against the
committed one. Exits non-zero if anything drifts.

The prediction parquets themselves are produced by the four domain adapters (domains/*/), which
require the original source repositories and a Kaggle account; they are provided as the benchmark
so this script reproduces every paper number without that setup.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
STEPS = [
    ("audit engine", "audit.engine"),
    ("shift diagnosis", "audit.diagnosis"),
    ("remediation ladder", "audit.remediation"),
    ("meta-analysis", "synthesis.meta"),
    ("tables", "synthesis.tables"),
    ("figures", "synthesis.figures"),
    ("taxonomy figure", "synthesis.fig_taxonomy"),
]


def main() -> int:
    t2_before = pd.read_csv(ROOT / "results" / "tables" / "T2_master.csv")
    for label, mod in STEPS:
        print(f"[run_all] {label} ...", flush=True)
        r = subprocess.run([sys.executable, "-m", mod], cwd=ROOT)
        if r.returncode != 0:
            print(f"[run_all] FAILED at {label}", file=sys.stderr)
            return 1
    t2_after = pd.read_csv(ROOT / "results" / "tables" / "T2_master.csv")
    if t2_before.equals(t2_after):
        print("[run_all] OK — master table reproduced identically.")
        return 0
    print("[run_all] WARNING — master table changed after re-run; inspect diff.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
