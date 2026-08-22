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
- **Minimum sample size, pre-registered before running:** a (state, year) cell is INCLUDED only if
  ALL of the following hold: (a) ≥ 500 total rows; (b) ≥ 30 rows in EACH of the two `race_black`
  subgroup categories (black, white) — the primary subgroup axis already used throughout this
  benchmark (`PRIMARY_AXIS["lending"] = "race_black"`, `audit/engine.py`); (c) ≥ 10 positive
  (approved) AND ≥ 10 negative (denied) outcomes overall, and the same ≥10/≥10 positive/negative
  minimum WITHIN each `race_black` subgroup separately (AUC is undefined with one class absent and
  numerically unstable with fewer than ~10 of either class). A cell failing ANY of these
  thresholds is EXCLUDED, never imputed, never merged into a neighboring cell, and never silently
  dropped without a recorded reason.
- Every excluded cell and its exact reason (which specific threshold failed, with the actual
  count) is written to `results/lending_excluded_cells.json` **before** any metric is computed on
  the included cells — this file is the audit trail for what "≈50 states" actually became after
  exclusion, and its row count is reported alongside the analysis, not left implicit.
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

**Primary within-domain analysis (the headline result), REVISED per user directive:** the
relationship between AUC$_{\text{dc}}$ (covariate-shift magnitude, treated as a **continuous**
variable) and the observed degradation pattern (ΔAUC, Δmacro-F1, ΔECE, ΔG). Tercile/quantile
binning is explicitly demoted to a **secondary visualization aid only** — tercile cut points are
sample-dependent, can hide a genuinely continuous relationship, and replacing one arbitrary scalar
threshold (the taxonomy's original AUC$_{\text{dc}}>0.65$ covariate cutoff) with another arbitrary
threshold (a tercile split) would undercut this paper's own central argument. The primary result is
therefore: (a) a **failure-fingerprint characterization** — for each of the four degradation axes,
whether/how strongly it varies with continuous AUC$_{\text{dc}}$, reported via a smooth
(e.g. LOWESS or binned-with-many-fine-bins-shown-as-a-continuous-curve) relationship, not three
coarse buckets; and (b) explicit reporting of whether the relationship is monotonic, flat, or
non-monotonic across the full observed AUC$_{\text{dc}}$ range, since a non-monotonic or flat
relationship is itself the finding the "magnitude does not predict damage" thesis needs. Tercile or
quantile bucketing may still be SHOWN as a secondary, clearly-labeled visualization for readability,
but is never the statistical basis for any claim.

**Secondary evidence:** the within-domain OLS slope of ΔG (or the chosen headline degradation
metric) on continuous AUC$_{\text{dc}}$, with the cluster-bootstrap CI above. This supersedes the
current `docs/INTERPRETATION.md`-only, n=3, unscripted "−0.099" slope (`docs/CLAIM_LEDGER.md`
C-31/M5) — once this analysis exists as a committed script output, the old number is retired, not
reported alongside it.

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
  train/holdout boundary), consistent with the split validated in Phase 1 (`security_provenance_audit.json`'s near-
  duplicate check found no evidence of excess cross-split duplication relative to within-split
  duplication -- this is not claimed as proof the split is leakage-free).
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
- **Reporting spec (mandatory at every severity level):** three recall numbers side by side, never
  just one — (i) **seen-family recall** (families still in the training pool at this severity),
  (ii) **unseen-family recall** (the families held out at this severity, evaluated in-source), and
  (iii) **overall attack recall** (pooled across seen+unseen, the number a practitioner monitoring
  aggregate detector performance would actually see). Reporting only the aggregate would repeat
  the exact failure mode Phase 1 found in the original manuscript (an aggregate number masking
  family-level collapse); reporting only per-family numbers would lose the practitioner-relevant
  "what does the dashboard show" framing. Both are required at every severity level, with exact
  binomial (Clopper–Pearson) CIs given the small per-family counts already visible in
  `security_label_space.csv`.
- **Negative control:** hold out a matched NUMBER of rows via random subsampling of a SINGLE
  already-well-represented family (e.g. remove an equivalent row count from `udp` alone, keeping
  all 5 families present) — same training-set-size reduction, no mechanism-coverage reduction.
  Expected direction (hypothesis, not guaranteed): this control should show little to no
  degradation relative to the full-coverage baseline, because it removes volume, not mechanism
  diversity — if it degrades comparably to the mechanism-holdout conditions, that would undercut
  the mechanism-coverage story and must be reported as such.

### S2 feasibility note (completed 2026-08-22 — pre-check required before writing any S2 code)

**Inspected `D:\Projects\ddos_xdomain_paper\src` as required before writing S2 code.** Findings,
against the exact five-point structure requested:

1. **Label definition:** `src/data.py::label_of()` maps every raw label to one of exactly three
   codes — `BENIGN=0`, `DDOS=1`, `OTHER_ATTACK=-1`. This is a **binary/ternary** task (attack vs.
   benign vs. excluded-as-out-of-scope), not a multiclass attack-family task. `family_of()`
   computes a family string (`"volumetric"`, `"reflection_amplification"`,
   `"low_rate_application"`, `"benign"`, `"other"`) but this is used **only for post-hoc,
   per-family evaluation reporting** — the model is never trained to predict family. This is the
   identical pattern TrustShift's own `domains/security/run_pipeline.py` already uses
   (`to_binary()` + `_family()` for post-hoc grouping).
2. **Class mapping:** family taxonomy is defined in `src/config.py::FAMILY_PATTERNS` /
   `FAMILY_PRECEDENCE` with an explicitly documented **longest-match-wins, precedence-tiebreak**
   algorithm — the docstring calls out, by name, the exact bug class TrustShift's own `_family()`
   is vulnerable to in principle: *"A first-family-wins rule would misfile 'DoS slowloris' as
   volumetric because 'dos' matches before 'slowloris' is ever tested."* Checked against
   TrustShift's actual `_family()` substring list (`domains/security/run_pipeline.py`): TrustShift
   does not include a bare `"dos"` pattern, so this specific failure mode does not fire for the
   labels currently in play (`"DoS Hulk"`, `"DoS GoldenEye"`, `"DoS slowloris"` all correctly
   resolve to `hulk`/`goldeneye`/`slowloris` — verified by hand against the substring list). This
   is a **near-miss, not a live bug** for the current dataset, but TrustShift's algorithm is less
   principled in general than the sibling project's, and should adopt the same longest-match +
   precedence approach before any finer-grained (more families, more edge cases) mechanism work in
   Phase 3, to avoid relying on having gotten lucky with the current label set.
3. **Train/test split:** `src/pipeline.py`'s `train_test_split(X, y, fam, test_size=TEST_SIZE,
   random_state=seed, stratify=y)` is a **plain random row-level split, stratified only by binary
   label** — it does NOT use capture-file/session grouping. TrustShift's Phase-1 native-file
   split validated by the Phase-1 duplicate check (`security_provenance_audit.json`) is *more*
   rigorous in this respect than this reference
   project in exactly the respect Phase 1 hardened.
4. **Whether held-out classes are genuinely absent from training:** not applicable as asked,
   because there is no multiclass task to hold classes out of — the reference project never
   constructs one either.
5. **Whether multiclass evaluation would be statistically meaningful:** cannot be assessed from
   this codebase because it does not attempt it; would need to be established fresh for
   TrustShift's own data (family-level `n` in `security_label_space.csv` — several families have
   `n < 700` rows in their native training files, e.g. netbios 644, portmap 685, which is thin for
   a from-scratch multiclass classifier).

**Verdict: S2 (genuine multiclass attack-family classification with truly held-out classes) is
marked NOT APPLICABLE for Phase 3.** The most directly comparable prior work by the same author
does not attempt it either, uses the identical binary + post-hoc-family-reporting pattern
TrustShift already has, and TrustShift's own per-family sample sizes are thin for several
families. Building S2 from scratch to force a distinction the field's own closest precedent avoids
would be exactly the kind of manufactured-evidence risk this design gate exists to prevent. **S1
(binary task, mechanism-coverage severity ladder) is the sole security experiment in Phase 3.**
The terminology distinction remains locked regardless: S1 measures **unseen attack-mechanism /
attack-family deployment shift in a binary benign-vs-attack task** — never label-space
expansion, never open-set classification, since no class is ever removed from or added to the
binary label space.

### Mechanism-category taxonomy: documented disagreement with prior work, not resolved by fiat

`ddos_xdomain_paper/src/config.py::FAMILY_PATTERNS` uses a **3-tier** scheme (`volumetric`
[includes syn, hulk, goldeneye, flood], `reflection_amplification`, `low_rate_application`).
`domains/security/provenance_audit.py::MECHANISM_CATEGORY` (Phase 1, this repo) uses a **4-tier**
scheme (`protocol_exhaustion` [syn alone], `reflection_volumetric`, `application_flood` [hulk,
goldeneye], `application_slow`) that separates protocol-layer and application-layer floods from
pure volumetric/reflection traffic. These are two different, both-defensible categorization
choices by the same author across two projects, and this document does not adjudicate between
them — that is an open reconciliation item for the user to decide before Phase 3 runs, not
something to resolve silently:
- **Argument for the 3-tier (ddos_xdomain_paper) scheme:** consistency with the author's existing,
  more mature, published-toward research artifact; coarser categories may be more robust to
  labeling ambiguity.
- **Argument for the 4-tier (provenance_audit.py) scheme:** separates mechanisms that operate at
  different OSI layers (TCP state-exhaustion vs. HTTP-layer floods vs. raw volumetric/reflection),
  which is the distinction Phase 1's finding actually turns on (source training data has zero
  application-layer representation under the 4-tier scheme; under the 3-tier scheme, Hulk/
  GoldenEye would count as "volumetric," which is also present in training via syn/udp, muddying
  that finding).
**This document documents the disagreement rather than picking a side; the mechanism-category
uncertainty itself is reported as uncertainty, not papered over.** Whichever scheme is chosen, the
severity ladder in the next section is defined in terms that work under either taxonomy (fraction
of trainable non-benign families withheld), so the choice does not block T3.2 execution — it
blocks only how the *manuscript* later describes why the mechanisms differ.

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

### Dataset-pair decision gate: HMDA+Security vs. HMDA+NLP, scored on validity criteria (not convenience)

Per user directive, the preference for lending+security is checked against lending+NLP using six
explicit criteria, before committing to either pair.

| Criterion | Lending+Security | Lending+NLP | Winner |
|---|---|---|---|
| All four mechanisms feasible (valid or defensible, not N/A) across the pair? | Yes — covariate: both valid; prior: both valid; concept: lending's natural-subpopulation construction defensible, security's questionable; **novel-class: security supplies a VALID cell by reusing S1** | Partial — covariate: both valid; prior: both valid; concept: lending's natural-subpopulation defensible, NLP's questionable; **novel-class: NEITHER dataset offers a valid construction** (lending N/A, NLP questionable/likely N/A) | **Security** (only pair with novel-class coverage) |
| No prohibited label manipulation (no arbitrary concept-shift flip)? | Compliant (natural-subpopulation approach) | Compliant (same approach could apply) | Tie |
| Shared model/features possible without retraining the reuse-only domains? | N/A — neither lending nor security is a reuse-only domain (`CLAUDE.md` hard rule #6 only restricts clinical/NLP); both can be resampled or re-scored with full flexibility | NLP IS reuse-only: covariate/prior constructions are capped to resampling a FIXED, already-scored example pool, limiting the achievable severity range compared to a domain with no such ceiling | **Security** (lending+security pair has no reuse-flexibility ceiling on either member) |
| Severity measurable on a comparable scale (AUC$_{\text{dc}}$)? | Yes, both members already compute a domain classifier | Yes, both members already compute a domain classifier | Tie |
| Reproducible (deterministic, seeded)? | Yes — both `domains/lending/train.py` and `domains/security/run_pipeline.py` are fully scripted and seeded | Yes — `audit/diagnosis.py::_nlp_text_features()` is deterministic/seeded too | Tie |
| Interpretable to a reader without excessive caveats? | High — tabular loan features (LTV, purpose) and network-flow mechanism categories are both intuitive; security's novel-class cell IS the paper's own headline finding, not a bolted-on addition | Lower — resampling a 50-dim TF-IDF/SVD embedding by propensity is harder to explain intuitively than resampling by LTV or by attack mechanism | **Security** |

**Decision: Lending + Security is selected**, not because it was the first idea, but because it
scores at least as well as Lending + NLP on every criterion and strictly better on three
(mechanism coverage — specifically novel-class, which Lending + NLP cannot supply at all;
reuse-flexibility; interpretability). NLP is retained as an **optional third dataset** for the two
cells it can support (covariate, prior/label) if additional cross-modality confirmation is wanted,
but it does not replace security in the core T3.3 design.

### Scope from the validity + dataset-pair gates together

**T3.3 scope: lending (4 cells: 2 valid, 1 natural-subpopulation, 1 N/A) + security (3 cells: 2
valid, 1 questionable/likely N/A, 1 reused-from-S1) + NLP as an optional third dataset (2 valid
cells only: covariate and prior/label).** Clinical is excluded from T3.3 entirely: every one of
its cells is questionable given the target-feature-unavailability constraint, and forcing an
NHANES-only construction would not test what T3.3 needs to test. This is NOT a rectangular 4×3×2
matrix — it is 8–10 valid/defensible cells out of a possible 16, reported honestly as such.

### Formal per-cell specification (exact template, every valid/defensible cell)

Per user directive, every cell entering the actual T3.3 experiment (not the questionable/N/A ones)
is specified as: **mechanism → intervention → invariant/held-fixed quantity → severity measure →
expected effect → failure condition**. "Failure condition" states what result would mean the
construction did NOT produce a valid test of that mechanism (as opposed to a result that simply
contradicts the hypothesis, which is a finding, not a construction failure).

| Cell | Mechanism | Intervention | Held fixed | Severity measure | Expected effect (hypothesis) | Failure condition (construction is invalid, not just non-confirmatory) |
|---|---|---|---|---|---|---|
| Lending-covariate | Covariate | Resample evaluation rows by propensity on {ltv, purpose_refi, aus_automated} | Fitted model, $P(Y\mid X)$, label definition | Resampling strength, calibrated to a target AUC$_{\text{dc}}$ against the unresampled set | Little discrimination/calibration degradation even at High severity (negative-control role) | If resampling collapses to near-zero effective sample size (analogous ESS check to the NLP concept probe) at any severity level, the construction has failed to actually shift the distribution and must be flagged, not silently accepted |
| Lending-prior | Prior/label | Subsample approved/denied within a fixed covariate stratum to hit a target $\pi$ | $P(X\mid Y)$ within each class (drawn unchanged from existing rows) | $\lvert\Delta\pi\rvert$ achieved | Operating-point performance (macro-F1) moves even at Low severity, per the Phase-1 finding | If the fixed covariate stratum is too small to reach the target $\pi$ at High severity without resampling with very heavy replacement (effectively duplicating rows), the construction is invalid at that severity level |
| Lending-concept (natural) | Concept | Identify the (year × stratum) cell where the FITTED model's real residual pattern already shifted most between 2020–21 and 2023–24 | Nothing synthetic — the model and data are both real and unmodified | Measured residual-shift magnitude (discovered, not dialed) | Discrimination/calibration degrade more than covariate/prior cells at comparable $\Delta\pi$/AUC$_{\text{dc}}$ | If no (year × stratum) cell shows a detectably different residual pattern from the rest of the panel (i.e. the panel is homogeneous), there is no natural concept-shift cell to use and this row is reclassified N/A rather than forced |
| Security-covariate | Covariate | Resample within `reflection_volumetric`-only rows by flow-duration/byte-count propensity | Attack mechanism/label, model | Resampling strength vs. target AUC$_{\text{dc}}$ | Little degradation (negative-control role, mirrors lending-covariate) | Same ESS-style collapse check as lending-covariate |
| Security-prior | Prior/label | Vary benign:attack ratio within a single fixed mechanism (e.g. `syn`-only) | Which mechanism is present, $P(X\mid Y{=}1)$ | $\lvert\Delta\pi\rvert$ achieved | Operating-point performance moves even at Low severity | If the fixed-mechanism subset is too small to vary prevalence meaningfully (see `security_label_space.csv`'s per-family n), flagged invalid at that severity |
| Security-novel-mechanism | Novel-mechanism (reused from S1) | Mechanism-coverage holdout, as specified in T3.2 | Binary task/label space | Fraction of trainable `reflection_volumetric` coverage withheld | Recall degrades with increasing severity (H2 in T3.2) | Covered by T3.2's own severity-ladder validity, not re-derived here |
| NLP-covariate (optional) | Covariate | Resample existing scored kaggle/reddit/twitter examples by TF-IDF/SVD-feature propensity | Model (frozen), label | Resampling strength vs. target AUC$_{\text{dc}}$, capped by the fixed example pool's size (reuse-only constraint) | Little degradation at low/moderate severity; may not reach High severity given the pool-size ceiling — reported as an explicit scope limitation, not hidden | If the fixed pool cannot reach even the Low severity target AUC$_{\text{dc}}$, this cell is reported as infeasible at that severity rather than forced |
| NLP-prior (optional) | Prior/label | Subsample existing examples to hit a target class ratio | $P(X\mid Y)$ within class (unchanged, existing rows) | $\lvert\Delta\pi\rvert$ achieved, capped by pool size | Operating-point performance moves at Low severity | Same pool-size ceiling check as NLP-covariate |

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
1. T3.1: the failure-fingerprint characterization against **continuous** AUC$_{\text{dc}}$ (not
   tercile-binned — terciles are a secondary visualization only, see above) — computed before any
   regression.
2. T3.2 (S1): per-severity-level (seen vs. held-out mechanism-family) recall and AUC on both the
   in-source holdout and the cross-dataset (CICIDS2017) target, compared against the matched
   negative control.
3. T3.3: per valid cell, ΔAUC, Δmacro-F1, ΔECE, ΔG at each severity level, summarized as which
   trustworthiness axis fails at which mechanism × severity combination (a fingerprint matrix, the
   same style as Table~\ref{tab:fingerprint} in the manuscript, extended with severity as a third
   dimension). This is the **descriptive** primary result.

**T3.3's separate inferential question (does not collapse the matrix into one p-value):**
*Does changing the shift mechanism produce different failure patterns after controlling for the
shift construction and severity?* Answered by comparing, at MATCHED severity (matched
AUC$_{\text{dc}}$ or matched $\lvert\Delta\pi\rvert$ as appropriate) across mechanisms: effect
size (the magnitude of ΔAUC/Δmacro-F1/ΔECE/ΔG), its bootstrap CI, its direction, and whether that
direction and magnitude are consistent across severities and across the datasets that share a
given mechanism cell (e.g. lending-covariate vs. security-covariate, both negative-control-role
cells). A single omnibus regression across all mechanisms/datasets/severities pooled together is
explicitly rejected as the primary inferential tool — it would average over exactly the
heterogeneity (Simpson-style clustering, already documented in `main.tex`'s Section~\ref{sec:meta}
box for the cross-domain M2 regression) this benchmark exists to expose. Regression remains
available as **secondary**, reported per-cell or per-mechanism, never pooled across mechanisms as
a single headline number.

**Secondary endpoints:**
1. T3.1: the within-domain OLS slope (ΔG or chosen metric on continuous AUC$_{\text{dc}}$) with
   cluster-bootstrap CI, reported as a supplement to the primary fingerprint, not as the headline.
2. T3.2: the negative control's degradation relative to the matched mechanism-holdout condition
   (a difference-in-differences style comparison, not a standalone claim).
3. T3.3: cross-dataset comparison of which mechanism produces the largest average degradation per
   axis, treated as descriptive, not as a formally tested hypothesis given the small number of
   valid cells; per-cell/per-mechanism regression slopes, never a single pooled slope.

**Statistical tests:**
- T3.1: cluster bootstrap (2,000 resamples, states as clusters) for all CIs; no p-value hunting
  across cells — the primary fingerprint is a proportion/count summary against a continuous
  covariate, not a hypothesis test.
- T3.2: exact binomial CI (Clopper–Pearson) on per-family recall at each severity level, given
  family-level sample sizes are frequently small (see `security_label_space.csv`); comparison
  between severity levels via a bootstrap difference-in-recall with the same 2,000-resample
  convention used throughout this benchmark (`config.N_BOOT`).
- T3.3: bootstrap CIs on each ΔAUC/Δmacro-F1/ΔECE/ΔG at each severity level (per-cell, not pooled
  across mechanisms), consistent with `audit/engine.py`'s existing bootstrap conventions
  (`N_BOOT=2000`, `BOOT_CAP=40000`).

**Per-experiment expected direction, stated as hypotheses (not guaranteed outcomes) — these are
operational sub-hypotheses, distinct from the three THESIS-level falsification criteria below:**
- h1 (T3.1): degradation on any axis will correlate more strongly with which subpopulation/year
  combination is evaluated than with AUC$_{\text{dc}}$ magnitude alone — i.e. the fingerprint's
  pattern will not be a smooth, tight function of continuous AUC$_{\text{dc}}$.
- h2 (T3.2, S1): recall on held-out `reflection_volumetric` families will degrade with increasing
  severity (more families withheld), while the negative control (matched volume reduction, no
  mechanism-coverage reduction) will show materially smaller degradation. This hypothesis can fail
  — if held-out families are detected as well as trained ones (plausible, since Phase 1 already
  found dns/ntp/snmp/tftp, never trained on, detected at ≥0.93 recall in-domain), that is itself a
  finding about within-category generalization, not a null result to be hidden.
- h3 (T3.3): covariate-shift cells (both negative-control directions) will show smaller
  discrimination/calibration degradation than prior/label-shift cells at matched severity,
  consistent with Phase 1's lending finding that ranking survives covariate shift better than the
  operating point survives prevalence shift.

---

## Falsification criteria for the TrustShift thesis (fixed now, per user directive, before any Phase 3 result exists)

These are distinct from h1–h3 above: h1–h3 are per-experiment operational expectations; H1–H3
below are the paper's actual scientific claims, with a decision rule fixed in advance so that a
non-confirmatory result cannot be quietly reframed as "an interesting finding" after the fact.

**H1 — mechanism produces distinguishable failure fingerprints** (different shift mechanisms lead
to different patterns of which trustworthiness axis fails).
- *Supported* if: at least two tested mechanisms show a qualitatively different fingerprint
  (different axis failing, or the same axis failing with non-overlapping bootstrap CIs on effect
  size) consistently across multiple datasets/severities — not one isolated cell.
- *Mixed* if: fingerprints differ, but inconsistently across datasets, or the difference is
  present but small/within-noise for some mechanism pairs.
- *Falsified* if: all tested mechanisms produce statistically indistinguishable fingerprints
  (overlapping CIs on every axis, no consistent qualitative pattern) across datasets.

**H2 — within a fixed mechanism, magnitude is not a sufficient predictor of failure** (knowing
severity/AUC$_{\text{dc}}$ alone does not determine which axis fails or by how much).
- *Supported* if: within a single mechanism's severity ladder, comparable severity levels (or
  comparable AUC$_{\text{dc}}$) produce detectably different failure patterns depending on
  domain/subpopulation/cell — i.e. within-mechanism variance in outcome is large relative to the
  severity-driven trend (the same logic that exposed the original cross-domain M2 regression as a
  Simpson artifact, now checked WITHIN a controlled mechanism rather than across uncontrolled
  domains).
- *Mixed* if: magnitude explains some but not all of the variance (e.g. a moderate $R^2$,
  consistent direction but incomplete).
- *Falsified* if: magnitude alone (continuous AUC$_{\text{dc}}$/severity) explains the observed
  failure pattern well (high $R^2$, consistent slope, tight CI) within a fixed mechanism — i.e.
  mechanism identity adds no information beyond magnitude once magnitude is controlled.

**H3 — the staged audit (Stage A screening + Stage B confirmation) provides more actionable
information than a scalar shift score alone.**
- *Supported* if: cases exist where two shift-points have similar scalar magnitude
  (AUC$_{\text{dc}}$) but different observed outcomes, and the staged diagnosis (mechanism
  category, or `inconclusive` with its stated reason) correctly anticipates which one fails and
  how — i.e. the diagnosis category, not just the AUC$_{\text{dc}}$ number, tracks the outcome.
- *Mixed* if: the staged diagnosis adds information in some domains/mechanisms but not others.
- *Falsified* if: scalar shift magnitude alone predicts the failure pattern as well as the full
  diagnosis does; the staged protocol's categories add no discriminative power beyond
  AUC$_{\text{dc}}$.

**Standing rule: the title and central claim are not protected from this outcome.** The current
working title — *"TrustShift: Shift Mechanism Is More Informative Than Shift Magnitude for
Machine-Learning Deployment Audits"* — is a claim to be tested by H1–H3, not a conclusion to be
defended regardless of what Phase 3 shows. If Phase 3 shows H2 falsified (magnitude, once
controlled within a fixed mechanism, explains the failure pattern well) or H1 falsified
(mechanisms are not distinguishable), the title and the paper's central claim change to match the
evidence. This applies symmetrically to the manuscript sections already deferred to Phase 4
(Discussion, Conclusion, Contributions, Figure 1, the title itself) — none of them are rewritten
around a story Phase 3 does not support.

---

## What this document does NOT do

It does not report a single Phase 3 result. It does not assume lending+security are sufficient
without the validity table above. It does not leave the lending bootstrap method as an open
choice. It does not call the security phenomenon "novel-class" or "label-space expansion." Any
Phase 3 code or manuscript text that contradicts a rule in this document should be treated as a
bug in the execution, not a reason to revise this document post hoc — revisions to this design,
if needed, require the same explicit user sign-off that produced it.
