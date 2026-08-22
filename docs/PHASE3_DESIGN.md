# Phase 3 Experiment Design — FROZEN before any Phase 3 result is generated

Status: design-only. No Phase 3 code has been run against this design. Per the user's explicit
instruction, this document is committed to git before T3.1/T3.2/T3.3 execute, so that later
results can be checked against a design that was fixed in advance, not adjusted after seeing
outcomes.

Everything in this document derives from artifacts already regenerated in Phase 1
(`docs/CLAIM_LEDGER.md`, `results/security_provenance_audit.json`,
`results/security_label_space.csv`, `results/lending_panel_provenance.json`,
`results/tables/T2_master.csv`). Nothing here is retrospectively fitted to a Phase 3 result,
because no Phase 3 result exists yet.

---

## T3.1 — Lending: fixed source model, many target slices

### Source model-selection rule (exact)

One frozen source model per seed, trained exactly as `domains/lending/train.py::_fit` already
does for the `lightgbm_temporal` experiment (train on 2020–21, `CAP_TRAIN=2,000,000` stratified
sample, `class_weight="balanced"`, `LGBMClassifier(n_estimators=400, learning_rate=0.05,
num_leaves=64, subsample=0.8, colsample_bytree=0.8)`). **No retraining per state.** Three seeds
already exist and will be reused: `SEEDS_NEW = [42, 7, 123]` (`config.py`). Each seed's model is
evaluated, unchanged, against every target cell below.

### Target-cell inclusion/exclusion rules (exact)

- Target cells = every (state × year) combination for year ∈ {2022, 2023, 2024}, state ∈ all
  `state_fips` values present in `features_panel.parquet` (`lending_panel_provenance.json`
  currently records `n_states`; exact count confirmed at T3.1 run time, expected ≈50).
- **Minimum sample size, pre-registered before running:** a (state, year) cell is INCLUDED only
  if it has ≥ 500 total rows AND ≥ 30 rows in EACH of the two `race_black` subgroup categories
  (black, white) — the primary subgroup axis already used throughout this benchmark
  (`PRIMARY_AXIS["lending"] = "race_black"`, `audit/engine.py`). A cell failing either threshold
  is EXCLUDED, not imputed or merged into a neighboring cell.
- Every excluded cell and its exact reason (row count, or which subgroup count failed) is written
  to `results/lending_excluded_cells.json` before any metric is computed on the included cells —
  this file is the audit trail for what "≈50 states" actually became after exclusion.
- income_quartile is NOT used as a minimum-sample gate (it is a secondary subgroup axis, always
  well-populated by construction of the quartile cut); race_black is the gating axis because it
  is the domain's stated primary axis and the one most likely to be sparse in a small state.

### Model fixed, only the evaluation slice varies

For each of the 3 seeds' frozen models, score every included (state, year) cell using the model's
own `FEATURES` and `inc_edges` from that seed's ORIGINAL fit (the same income-quartile edges used
for the seed's `source_test`/`target_2022-24` evaluation already in the released benchmark — no
new quantile cut per state). Record per cell: AUC$_{\text{dc}}$ (domain classifier, source vs. this
cell's features), $\Delta\pi$, ΔAUC, Δmacro-F1, ΔECE, ΔBrier, ΔG (race_black), all against that
seed's own `source_test` (2020–21 heldout), exactly as `audit/engine.py` already computes for the
existing temporal targets.

### Dependence / uncertainty method (exact, resolved — not left open)

**Primary method: cluster (block) bootstrap over states, with year and seed as repeated factors
within each resampled state.** Procedure:
1. Resample the set of INCLUDED states with replacement, drawing the same number of states as are
   included (≈50, exact number from the exclusion log).
2. For each resampled state, include ALL of its eligible (year × seed) observations — i.e. every
   year (2022/2023/2024) and every seed (42/7/123) that state passed the inclusion gate for. This
   preserves the natural within-state year-clustering and does not synthetically balance years.
3. Recompute the quantity of interest (see primary endpoint below) on this resampled panel.
4. Repeat 2,000 times (`N_BOOT`, `config.py`); report the 2.5th/97.5th percentile as the 95% CI.
This is a standard cluster bootstrap (resampling the cluster unit — the state — with replacement,
keeping all of its natural sub-observations together) and is chosen over: (a) plain i.i.d.
bootstrap over (state,year,seed) triples, which would treat observations from the same state in
different years as independent when they plausibly share state-level confounders (regulatory
environment, economic conditions); (b) a mixed-effects model, which would require distributional
assumptions on the random effects that are not yet justified for this panel and adds a second,
un-pre-registered modeling choice. The cluster bootstrap is the pre-specified, sole primary method;
no alternative method is substituted after seeing results.

### Primary vs. secondary analysis

**Primary within-domain analysis (the headline result):** a **failure-fingerprint characterization**
of the relationship between AUC$_{\text{dc}}$ (covariate-shift magnitude) and the observed
degradation pattern (ΔAUC, Δmacro-F1, ΔECE, ΔG) — reported first as a qualitative/categorical
summary (e.g. "of N included cells, what fraction show discrimination failure / operating-point
failure / calibration failure / subgroup failure, and does this fraction vary with AUC$_{\text{dc}}$
tercile") **before** any regression line is fit. This directly answers the research question
(mechanism vs. magnitude) without presupposing a linear relationship.

**Secondary evidence:** the within-domain OLS slope of ΔG (or the chosen headline degradation
metric) on AUC$_{\text{dc}}$, with the cluster-bootstrap CI above. This supersedes the current
`docs/INTERPRETATION.md`-only, n=3, unscripted "−0.099" slope (`docs/CLAIM_LEDGER.md` C-31/M5) —
once this analysis exists as a committed script output, the old number is retired, not reported
alongside it.

### Analysis code paths

- `domains/lending/phase3_shiftpoints.py` (new): builds the (state, year, seed) score table,
  applies the inclusion gate, writes `results/lending_excluded_cells.json` and
  `results/lending_shiftpoints.csv`.
- `synthesis/phase3_lending_analysis.py` (new): cluster bootstrap, fingerprint summary, secondary
  slope+CI; writes `results/lending_phase3_analysis.json` and the severity-surface figure.

---

## T3.2 — Security: S1 (mandatory) and S2 (conditional)

### Terminology (locked, per user directive — do not deviate without explicit sign-off)

The observed phenomenon is described as **novel attack-mechanism / unseen attack-family
deployment shift**, explicitly distinguished from **label-space expansion / open-set
classification**. The binary task (benign vs. attack) is unchanged in both S1 and the existing
benchmark; what changes is the conditional feature structure of the positive class, because
deployment traffic uses attack mechanisms absent from training. This phrase, not "novel-class
shift," is used everywhere until/unless S2 produces a clean multiclass result that would justify
a genuine novel-class claim (see S2 below).

### Mechanism categories (defined BEFORE running any severity analysis)

From `results/security_provenance_audit.json` (`domains/security/provenance_audit.py`), using the
standard three-tier DDoS taxonomy (volumetric/reflection-amplification, TCP protocol-exhaustion,
application-layer; citation to be verified before manuscript use — the categorical structure
itself is standard security-domain knowledge, not fitted to this dataset's results):

| Category | Families (this dataset) | Present in source TRAINING files? |
|---|---|---|
| `reflection_volumetric` | dns, ldap, mssql, netbios, ntp, snmp, tftp, portmap, udp (udp bucket also absorbs udplag-labeled rows — see collision note below) | dns/ntp/snmp/tftp: **no** (testing-file only); ldap/mssql/netbios/portmap/udp: **yes** |
| `protocol_exhaustion` | syn | yes |
| `application_flood` | hulk, goldeneye, ddos (target-side, LOIC/HOIC-generated) | **no — entirely absent from source** |
| `application_slow` | slowloris, slowhttp | **no — entirely absent from source** |
| `benign` | benign | yes |

**Label collision, must not be conflated:** the string label `ddos` denotes two different
populations. In CIC-DDoS2019 (source) it is a residual/catch-all bucket, n=51 raw rows, negligible.
In CICIDS2017 (target) it is the dataset's own LOIC/HOIC-generated volumetric flood label, n=128,014
raw rows — a real, large, well-defined attack population. `ddos` is excluded from the family-overlap
count for this reason; it is not evidence of a shared, trained-on mechanism.

**Hard scope constraint, stated plainly rather than glossed over:** CIC-DDoS2019 as loaded here
contains **zero** `application_flood` or `application_slow` representation. This means S1 (a
within-source-dataset holdout) can vary coverage only WITHIN `reflection_volumetric` +
`protocol_exhaustion` — it cannot construct a true application-layer holdout from source data alone,
because that mechanism category does not exist in the source dataset at all. The severity ladder
below is designed around this real constraint, not around an idealized one.

### S1 (mandatory): within-CIC-DDoS2019 mechanism-coverage holdout

- **Unit of holdout: mechanism category coverage, not arbitrary individual families.** At each
  severity level, hold out an increasing FRACTION of the available `reflection_volumetric` family
  diversity from training (never `protocol_exhaustion`'s only family, `syn`, since removing the
  sole representative of a whole category is a different manipulation from reducing diversity
  within a well-populated category).
- **Native file/session grouping preserved throughout:** training/holdout assignment operates on
  whole native `-training` files (never splits a single capture file's rows across the
  train/holdout boundary), consistent with the leakage-safe split already validated in Phase 1
  (`security_provenance_audit.json`'s near-duplicate check).
- **Severity definition:** severity = (number of `reflection_volumetric` families excluded from
  training) / (total `reflection_volumetric` families available in training files: ldap, mssql,
  netbios, portmap, udp — 5 families, since dns/ntp/snmp/tftp are already testing-file-only and
  not part of the trainable pool). Three levels:
  - **Low severity:** hold out 1 of 5 families (20% coverage reduction).
  - **Medium severity:** hold out 2 of 5 families (40%).
  - **High severity:** hold out 3 of 5 families (60%).
  At each level, evaluate on (a) the held-out families themselves (in-source novel-family
  generalization) and (b) the full CICIDS2017 target (cross-dataset novel-mechanism
  generalization) — giving two comparison points per severity level: a within-source-mechanism-
  family holdout and the existing cross-dataset mechanism-category holdout.
  Family choice at each level is fixed in advance (not selected after seeing results): drawn in a
  fixed order by ascending native-training-file row count (smallest first), so severity increases
  monotonically with which specific families are held out, and the choice is reproducible from a
  rule rather than cherry-picked.
- **Negative control:** hold out a matched NUMBER of rows via random subsampling of a SINGLE
  already-well-represented family (e.g. remove an equivalent row count from `udp` alone, keeping
  all 5 families present) — same training-set-size reduction, no mechanism-coverage reduction.
  Expected direction (hypothesis, not guaranteed): this control should show little to no
  degradation relative to the full-coverage baseline, because it removes volume, not mechanism
  diversity — if it degrades comparably to the mechanism-holdout conditions, that would undercut
  the mechanism-coverage story and must be reported as such.

### S2 (conditional on a clean construction): multiclass attack-family classification

- **Pre-check before building anything:** read `D:\Projects\ddos_xdomain_paper\src` for existing
  multiclass label handling, splits, and any documented caveats about label cleanliness, BEFORE
  writing new code.
- **Go/no-go criteria (checked before any S2 result is generated, not after):**
  1. Every family has ≥ 200 rows in its native training file (else that class is not learnable at
     the granularity being asked of it).
  2. No family's rows are drawn from a single, tiny native file also used for something else
     (checked against `security_provenance_audit.json`'s file-level row counts).
  3. Class definitions map 1:1 to the `_family()` function already used throughout this benchmark
     (no redefinition of family boundaries specifically to make S2 work).
  4. The multiclass task is evaluated with the SAME native `-training`/`-testing` file split as
     the binary task (no new, more favorable split invented for S2).
  - **If any criterion fails, S2 is marked not applicable and the binary S1 result stands, with
    precise terminology (novel attack-mechanism / unseen attack-family), not forced into a
    multiclass framing.**
- **If S2 is clean:** it can distinguish true novel-class shift (a held-out class never seen in
  any form during training) from the novel-mechanism/subpopulation framing S1 uses (a category
  underrepresented but the binary task itself unchanged) — this distinction, if it survives the
  go/no-go check, becomes part of the Phase 4 taxonomy; if not, the taxonomy keeps only the S1
  framing.

### Analysis code paths

- `domains/security/phase3_mechanism_holdout.py` (new): builds the S1 severity ladder (3 levels +
  negative control), reusing `harmonize`/`to_binary`/`_family`/`MECHANISM_CATEGORY` from
  `run_pipeline.py` and `provenance_audit.py`.
- `domains/security/phase3_multiclass_check.py` (new): runs the S2 go/no-go check FIRST and exits
  with a written verdict (`results/security_s2_verdict.json`) before attempting any multiclass
  training.

---

## T3.3 — Compact controlled mechanism × severity study: validity-first, not matrix-first

Per user directive, this section evaluates whether each dataset × mechanism cell has a defensible,
non-artificial construction BEFORE committing to a rectangular design. Each valid cell gets a
one-line construction spec: `mechanism → what changes → what is held fixed → why defensible`.

**Hard constraint respected throughout:** clinical and NLP source models are REUSED, never
retrained (`CLAUDE.md` hard rule #6). Any cell touching clinical/NLP must be constructible by
resampling or re-scoring EXISTING predictions/features, never by fitting a new model.

### Validity table

| Dataset | Covariate shift | Prior/label shift | Concept shift | Novel-class shift |
|---|---|---|---|---|
| **Lending** | **VALID.** Resample the evaluation set by feature-propensity (e.g. toward high-LTV/refi-purpose loans) at controlled severity, holding the fitted model and its learned $P(Y\mid X)$ fixed → construction: reweight/resample an existing target year's rows by a propensity score on {ltv, purpose_refi, aus_automated}; fixed: model, label definition, all other features' conditional distribution; defensible: standard covariate-shift-by-propensity-resampling construction (Sugiyama-style), applied to an existing eval set, no retraining. | **VALID.** Subsample approved/denied at a controlled target ratio from within a FIXED covariate region (e.g. within a fixed income/purpose stratum); changes: $P(Y)$ only; fixed: $P(X\mid Y)$ within each class (sampled from its own natural conditional distribution, unchanged); defensible: standard label-shift-by-class-resampling construction. | **QUESTIONABLE, resolved as: use a NATURAL subpopulation, not a synthetic flip.** A synthetic label-flip rule was explicitly rejected by the user as making the expected failure trivial. Construction: identify the (year × stratum) cell in the existing panel where the FITTED model's residual/calibration pattern already changed the most between 2020–21 and 2023–24 (an empirically-discovered concept-shift-dominant slice), and treat that slice as the concept-shift condition; fixed: nothing is synthetically altered, the "shift" is the model's actual measured behavior on a real subpopulation; defensible: this is honest use of real heterogeneity already in the panel rather than an invented rule, but it means severity is discovered, not dialed — see severity note below. | **N/A.** Binary approve/deny has no natural analogue of an unseen class; forcing one would require inventing a category that does not exist in mortgage underwriting decisions. Marked not applicable, not forced. |
| **Security** | **VALID.** Within a FIXED, already-seen mechanism category (e.g. `reflection_volumetric` only), resample by flow-duration/byte-count propensity; fixed: attack mechanism/label, model; defensible: same propensity-resampling logic as lending, applied within one mechanism category so the label-conditional structure is untouched. | **VALID.** Within a FIXED, already-seen mechanism (e.g. `syn` only), vary the benign:attack ratio at controlled severities; fixed: which attack mechanism is present, $P(X\mid Y{=}1)$; defensible: does not touch which mechanism is present, purely a prevalence manipulation. | **QUESTIONABLE, likely N/A.** Would require two variants of the SAME nominal attack label generated by different tools/parameters with different feature signatures. Not confirmed to exist in the raw CIC-DDoS2019 files as loaded; requires a targeted check of file/tool metadata before deciding. If unavailable, mark N/A rather than synthetically perturb features to fabricate a concept shift. | **VALID, but this is S1/S2 — not a new construction.** T3.3's "novel-class" cell for security reuses the S1 severity ladder directly rather than building a redundant parallel experiment. |
| **NLP** | **VALID (feasible without retraining).** `_nlp_text_features()` already extracts TF-IDF/SVD features for existing kaggle/reddit/twitter examples; resample within an existing platform's examples by feature-propensity; fixed: model (frozen, scored predictions already exist), label; defensible: resampling of already-scored examples, no retraining, consistent with the reuse-only constraint. | **VALID (feasible without retraining).** Subsample existing platform examples to hit a target class-prevalence ratio; fixed: $P(X\mid Y)$ within class (drawn from existing examples unchanged); defensible: same resampling-only logic. | **QUESTIONABLE.** As with lending, would need a natural (not synthetic) concept-shift-dominant subpopulation among existing platform examples — feasible in principle (compare per-subpopulation residuals across platforms already scored) but not yet checked for a clean candidate; requires the same natural-subpopulation search as lending's cell before being called valid. | **QUESTIONABLE/likely N/A.** The 4-class label space (anxiety/depression/normal/stress) is fixed; no natural 5th class exists in the released predictions. Would need to invent one, which is exactly what the user's concept-shift warning cautions against applying elsewhere too. Provisionally N/A. |
| **Clinical** | **QUESTIONABLE.** Target-side (BRFSS) features are unrecoverable (documented limitation, Section~\ref{sec:limitations}); a covariate-shift construction could only resample WITHIN NHANES (source) itself, simulating a shift and evaluating on a held-out NHANES slice — feasible without retraining, but does not use real BRFSS data, so it tests "shift robustness in principle," not this domain's actual deployment shift. Marked questionable, not valid, until this scoping is confirmed acceptable. | Same constraint as covariate: source-side-only construction possible, not confirmed valuable given the domain's real limitation is target-feature unavailability, which this construction does not fix. Questionable. | **QUESTIONABLE**, same natural-subpopulation logic as lending/NLP would apply if pursued; not evaluated further given the domain's already-constrained feature access. | **N/A.** Diabetes risk score has no unseen-class analogue. |

### Decision from this table

**Recommended T3.3 scope: lending (4 cells: 2 valid, 1 natural-subpopulation, 1 N/A) + security (3
cells: 2 valid, 1 questionable/likely N/A, 1 reused-from-S1) + NLP as an optional third dataset (2
valid cells only, covariate and prior/label — concept and novel-class both questionable/N/A for
NLP).** Clinical is excluded from T3.3 entirely: every one of its cells is questionable given the
target-feature-unavailability constraint, and forcing an NHANES-only construction would not test
what T3.3 needs to test. This is NOT a rectangular 4×3×2 matrix — it is 8–10 valid/defensible
cells out of a possible 16, reported honestly as such.

### Severity levels (what "severity" means per mechanism, stated before running anything)

| Mechanism | Severity quantity | Low / Medium / High construction |
|---|---|---|
| Covariate (lending, security-within-category, NLP) | Propensity-resampling strength (KL divergence of the resampled feature distribution from the natural one, or equivalently the resulting AUC$_{\text{dc}}$ against the unresampled set) | Target AUC$_{\text{dc}} \approx 0.60$ / $0.75$ / $0.90$ against the natural distribution, achieved by tuning the propensity-resampling temperature — NOT by an arbitrary resampling fraction, so severity is measured on the same scale already used throughout this benchmark. |
| Prior/label (lending, security-within-category, NLP) | $\lvert\Delta\pi\rvert$ achieved by the resampling | Low = 0.10, Medium = 0.25, High = 0.40 (chosen to span the range already observed naturally in this benchmark: lending's natural $\Delta\pi \approx 0.02$–$0.08$, NLP's natural $\Delta\pi$ up to 0.52). |
| Concept (natural-subpopulation cells) | Not experimentally dialed — severity is whatever the discovered natural subpopulation exhibits; report the ACTUAL severity found (e.g. as a residual-shift magnitude) rather than forcing it into low/medium/high, and say so explicitly rather than pretending three controlled levels exist for this mechanism. | N/A by construction — this is the one mechanism in T3.3 that is discovered, not dialed, per the "no arbitrary label-flipping" directive. |
| Novel-mechanism (security S1) | Fraction of mechanism-category coverage withheld (already specified in T3.2 above: 20% / 40% / 60% of the 5 trainable `reflection_volumetric` families) | As specified in T3.2. |

### Negative controls (T3.3-level, in addition to S1's own negative control above)

1. **Large covariate shift, $P(Y\mid X)$ deliberately preserved:** the covariate-resampling
   construction above, by design, does not touch $P(Y\mid X)$ — evaluate it explicitly as the
   negative control for "large marginal shift ≠ damage," expected (hypothesis) to show little
   discrimination/calibration degradation even at High severity.
2. **Small marginal shift, label prior deliberately altered:** the prior/label-resampling
   construction at Low severity (small $\lvert\Delta\pi\rvert$) is expected (hypothesis) to still
   move operating-point performance (macro-F1) measurably, per the T1.1 lending finding that
   prevalence changes move the operating point even when ranking metrics barely move — this tests
   whether "small shift = safe" is also false, symmetric to control 1.

---

## Primary and secondary endpoints (all of Phase 3)

**Primary endpoints:**
1. T3.1: the failure-fingerprint characterization (fraction of cells failing on each axis, by
   AUC$_{\text{dc}}$ tercile) — categorical, computed before any regression.
2. T3.2 (S1): per-severity-level (seen vs. held-out mechanism-family) recall and AUC on both the
   in-source holdout and the cross-dataset (CICIDS2017) target, compared against the matched
   negative control.
3. T3.3: per valid cell, ΔAUC, Δmacro-F1, ΔECE, ΔG at each severity level, summarized as which
   trustworthiness axis fails at which mechanism × severity combination (a fingerprint matrix, the
   same style as Table~\ref{tab:fingerprint} in the manuscript, extended with severity as a third
   dimension).

**Secondary endpoints:**
1. T3.1: the within-domain OLS slope (ΔG or chosen metric on AUC$_{\text{dc}}$) with cluster-
   bootstrap CI, reported as a supplement to the primary fingerprint, not as the headline.
2. T3.2: the negative control's degradation relative to the matched mechanism-holdout condition
   (a difference-in-differences style comparison, not a standalone claim).
3. T3.3: cross-dataset comparison of which mechanism produces the largest average degradation per
   axis, treated as descriptive, not as a formally tested hypothesis given the small number of
   valid cells.

**Statistical tests:**
- T3.1: cluster bootstrap (2,000 resamples, states as clusters) for all CIs; no p-value hunting
  across cells — the primary fingerprint is a proportion/count summary, not a hypothesis test.
- T3.2: exact binomial CI (Clopper–Pearson) on per-family recall at each severity level, given
  family-level sample sizes are frequently small (see `security_label_space.csv`); comparison
  between severity levels via a bootstrap difference-in-recall with the same 2,000-resample
  convention used throughout this benchmark (`config.N_BOOT`).
- T3.3: bootstrap CIs on each ΔAUC/Δmacro-F1/ΔECE/ΔG at each severity level (per-cell, not pooled
  across mechanisms), consistent with `audit/engine.py`'s existing bootstrap conventions
  (`N_BOOT=2000`, `BOOT_CAP=40000`).

**Expected direction, stated as hypotheses (not guaranteed outcomes):**
- H1 (T3.1): degradation on any axis will correlate more strongly with which subpopulation/year
  combination is evaluated than with AUC$_{\text{dc}}$ magnitude alone — i.e. the fingerprint's
  categorical pattern will not be a smooth function of AUC$_{\text{dc}}$.
- H2 (T3.2, S1): recall on held-out `reflection_volumetric` families will degrade with increasing
  severity (more families withheld), while the negative control (matched volume reduction, no
  mechanism-coverage reduction) will show materially smaller degradation. This hypothesis can fail
  — if held-out families are detected as well as trained ones (plausible, since Phase 1 already
  found dns/ntp/snmp/tftp, never trained on, detected at ≥0.93 recall in-domain), that is itself a
  finding about within-category generalization, not a null result to be hidden.
- H3 (T3.3): covariate-shift cells (both negative-control directions) will show smaller
  discrimination/calibration degradation than prior/label-shift cells at matched $\lvert\Delta\pi
  \rvert$-equivalent severity, consistent with Phase 1's lending finding that ranking survives
  covariate shift better than the operating point survives prevalence shift.

---

## What this document does NOT do

It does not report a single Phase 3 result. It does not assume lending+security are sufficient
without the validity table above. It does not leave the lending bootstrap method as an open
choice. It does not call the security phenomenon "novel-class" or "label-space expansion." Any
Phase 3 code or manuscript text that contradicts a rule in this document should be treated as a
bug in the execution, not a reason to revise this document post hoc — revisions to this design,
if needed, require the same explicit user sign-off that produced it.
