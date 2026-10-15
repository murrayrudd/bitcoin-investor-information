# Data dictionary

All investor populations and records are synthetic. The archives contain no empirical investor microdata. Monetary and Bitcoin quantities use normalized model units. Individual market price responses are `100 × (P19_transfer / P19_control − 1)`. Response levels are percentages; response differences, standard errors, allowances, and interval widths are percentage points (pp).

## Identifiers

| Field | Meaning |
|---|---|
| `bank` | 0 = paper procedure 1; 1 = paper procedure 2. Each uses an independently generated training bank. |
| `panel` | O1 = holdings; O2 = holdings and beliefs; O6 = complete linked modules; O7 = the same complete modules without cross-module investor linkage. |
| `case_id` | Zero-based main-run parameter-setting identifier, 0–255. |
| `replicate` | Evaluation market within a setting, 0–63. |
| `component` | `mixed`, `belief`, `funding`, `allowance`, `faces`, or `corners`. |
| `group` | A reported component or `primary`, the declared weighted mixture of the first four components. |
| `x` | Nine-dimensional normalized parameter coordinate in [−1, 1]. The specification gives parameter names and physical bounds. |
| `comparison`, `expanded` | Baseline and richer/linked information designs in paired contrasts. |

## Main summary records

`panel_summary.csv` has one row per group, bank, panel, and metric. `estimate` is the group estimate. `mcse` is its bootstrap Monte Carlo standard error; `ci95_low` and `ci95_high` are percentile limits across 2,000 setting resamples. `bootstrap_valid` is the number of finite bootstrap values. Missing bootstrap entries are retained for descriptive metrics not assigned a bootstrap interval.

Coverage and empty-rate metrics are **fractions** in the main summaries. Multiply by 100 to report percentages. Some preliminary summary files instead use explicit `_pct` columns. Never infer units solely from a value's magnitude.

`coverage` checks whether the response interval contains the independent reference mean. `definite_coverage` requires the full 99% interval around that mean to be contained. `possible_coverage` requires overlap with that reference interval. These labels express sensitivity to reference Monte Carlo error; they are not separate empirically validated coverage guarantees.

`true_parameter_retention` records acceptance of the generating parameter. It is distinct from response coverage. `mean_width_pp` uses nonempty intervals; `n_nonempty` gives their count. Setting-level tables retain both width sums divided by all 64 evaluation markets and conditional widths so group summaries can handle empty intervals consistently.

`paired_contrasts.csv` reports jointly nonempty-pair fractions, paired width reductions in pp, and paired relative reductions in percent. Positive reductions indicate narrower intervals under the expanded or linked design. `setting_contrasts.csv` retains the quantities used before setting aggregation.

`reference_means.csv` contains each setting's independent mean response, its standard error, and its 99% interval from 1,024 reference markets. The accompanying `evaluation_mean_response_pp` comes from a separate 64-market evaluation sample and is not used as the coverage target.

`primary_intervals.csv` contains bank, panel, setting, replicate, emptiness, lower and upper endpoints, width, response allowance, feasibility ratio, and coverage/retention flags. Empty records retain missing endpoint values; they are not replaced with successful fits. See the filename caveat in `RESULTS_MAP.md`.

`endpoint_audits.csv` reports selected fitted endpoint responses, allowances, fresh reference means and standard errors, 99% intervals, and observation-score retention at those locations. Audit failures and missing endpoints are retained. `search_checks.csv` records expanded search diagnostics. `integrity.json` records aggregate counts, residual maxima, optimizer status counts, bootstrap settings, and completion status.

## Archived arrays and fitted objects

| Main-run directory | Contents |
|---|---|
| `evaluation/` | Chunked NPZ records for 16,384 evaluation markets; response arrays, seeds, observation-feature arrays, parameters, and diagnostics |
| `reference/` | Chunked NPZ records for 262,144 independent reference markets |
| `fits/` | JSON records for primary interval fits and additional search checks |
| `models/` | Eight frozen NPZ model objects: two banks × four information designs |
| `audit/` | Endpoint design and 144 NPZ files, each with 128 fresh audit markets |
| `summary/` | CSV summaries and integrity/verification JSON records |

Load NPZ files with `numpy.load(path, allow_pickle=False)`. Their `metadata` entry is a JSON string stored as a NumPy string scalar. Data arrays and recorded seeds remain numeric. Model objects include polynomial coefficients, covariance-whitening transforms, response coefficients, cutoff coefficients, training coordinates, and metadata. Final cutoff multipliers modify observation acceptance; the inherited response allowances remain fixed.

The calibration attachment has three roots: `results/expanded_pilot/`, `results/repaired_pilot/`, and `results/continuation_v05/`. Their `specification.json`, model metadata, stage protocols, and column names define their stage-specific structures. Earlier designs and pilot counts must not be treated as the final main-run allocation.

## Provenance and omission rules

The original main archive omitted raw investor releases but retained the summaries used for inference, all fit records, models, and seeds. The full route reconstructs those releases. The development attachment retains its existing numerical banks and raw releases.

Packaging removes manuscript documents, prose-generation files, document-check records, and redundant historical plot exports. No numerical simulation, fitting, calibration, or audit record was intentionally removed. Exact excluded filenames within the result trees are listed in `release-assets.json`. Archive hashes identify the complete retained attachments.
