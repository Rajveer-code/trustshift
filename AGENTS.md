# TrustShift — project rules

**Read `PLAN.md` first. It is the single source of truth.** Resume = execute the first unchecked task in PLAN.md §5. No re-planning, no re-asking for files already referenced there.

## What this is

Cross-domain benchmark + paper: one audit protocol (discrimination, calibration, subgroup reliability, significance, shift diagnosis) applied to four real deployment shifts (clinical NHANES→BRFSS, mental-health NLP Kaggle→Reddit/Twitter, lending HMDA temporal/geographic, security CIC-DDoS2019→CICIDS2017). Target: Applied Intelligence (Springer, subscription route, $0) by ~2026-09-15. Sole author: Rajveer Singh Pall.

## Hard rules

1. **No AI attribution** in commits, code, or paper. Ever.
2. **No invented numbers.** Every paper number traces to a file in `results/`. Unknown → `[PLACEHOLDER: how to get]`.
3. Verify each PLAN.md task's Verify step before checking its box; show output.
4. All paths/constants come from `config.py`. Never hardcode paths in scripts.
5. Seeds fixed (config.py). Bootstrap N=2000. BH-FDR q=0.05.
6. Predictions from the clinical and NLP source repos are REUSED as data (never retrain those); lending and security models are trained here.

## Commands

```powershell
# environment (venv at .\venv)
.\venv\Scripts\python.exe -m pytest tests/ -q       # run before any commit
.\venv\Scripts\python.exe -m domains.clinical.adapter
.\venv\Scripts\python.exe -m domains.nlp.adapter
.\venv\Scripts\python.exe -m domains.lending.train
.\venv\Scripts\python.exe -m domains.security.run_pipeline
.\venv\Scripts\python.exe -m audit.engine           # writes results/audit_{domain}.json
.\venv\Scripts\python.exe -m synthesis.tables
.\venv\Scripts\python.exe -m synthesis.figures
```

## PATHS

See the single PATHS block in `config.py`. Source repos (read-only from here):
- clinical: `D:\Projects\diabetes_prediction_project\federated`
- nlp: `D:\Projects\mental-health-fairness-nlp-main`
- lending data: `D:\Projects\CATE-HMDA-Heterogeneous-Effects\data\`
- fairscope lib: `D:\Projects\fairscope` (installed editable)

Never modify the source repos; copy logic, not files, and note provenance in docstrings.
