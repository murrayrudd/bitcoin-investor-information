# Map from the working paper to retained results

This map refers to manuscript version 2, October 15, 2026. The final Monte Carlo uses experiment 0.6.0. Preliminary calibration and validation use 0.5.0 and earlier development records. Do not substitute preliminary-validation coverage or widths for main-run results.

Paths beginning `results/` refer to an extracted work directory. The main summary files are also copied into the repository's `reference/` folder, except the large interval table. `quick` regenerates the main files; `verify` also regenerates the preliminary summary files.

| Paper location | Scientific source | Selection or calculation |
|---|---|---|
| Table 1: parameter ranges | `results/confirmation_v06/specification.json` | `parameters`, `bounds`, and `config` |
| Table 2: information designs | `engine/pilot/observations.py`; `engine/pilot/inference_features.py` | Permitted fields and derived summaries for O1, O2, O6, O7 |
| Figure 1 and Table B1: design | Main `specification.json`; `reference/integrity.json` | Settings, component weights, market counts, and interval counts |
| Table 3: primary coverage and widths | `reference/panel_summary.csv` | `group=primary`; metrics `coverage`, `empty_rate`, `mean_width_pp` |
| Figure 2A: widths | `reference/panel_summary.csv` | `group=primary`, `metric=mean_width_pp`; estimates and `ci95_*` |
| Figure 2B and Table C10: primary gains | `reference/paired_contrasts.csv` | `group=primary`; width reductions in percentage points and percent |
| Table C1: final cutoff multipliers | `results/continuation_v05/calibration_results.json` | Eight model multipliers and order-statistic calibration records |
| Table C2: preliminary gates | `results/continuation_v05/specification.json`; its `summary/` CSVs | Declared gates, retention, response coverage, search checks, and endpoint audits |
| Table C3: earlier weak locations | `results/continuation_v05/summary/retention_by_case.csv` | Selected focal cases and binomial lower bounds |
| Table C4: preliminary widths/coverage | `results/continuation_v05/summary/intervals_by_scope.csv` | `scope=all`; 576 fitted validation releases per procedure–design pair |
| Table C5: reference responses | `reference/reference_means.csv` | Component means/ranges and reference standard errors; primary weights below |
| Table C6: component performance | `reference/panel_summary.csv` | Mixed, belief, funding, allowance, faces, and corners groups |
| Table C7: component information gains | `reference/paired_contrasts.csv` | Component groups; `metric=width_reduction_percent` |
| Table C8: endpoint-search checks | `reference/search_checks.csv`; `reference/integrity.json` | Enlarged searches, endpoint movements, empty-set recoveries, and residuals |
| Table C9: response audits | `reference/endpoint_audits.csv` | 144 selected endpoints; each has 128 fresh audit markets |
| Interval-level investigation | `results/confirmation_v06/summary/primary_intervals.csv` | All 131,072 primary-method records, including stress settings |
| Setting-level investigation | `reference/setting_summary.csv`; `reference/setting_contrasts.csv` | Performance and paired quantities before aggregation across settings |

The historical filename `primary_intervals.csv` means **primary search method**, not only the primary sample. Filter `component` to `mixed`, `belief`, `funding`, and `allowance` for the 98,304 primary-sample intervals. Keep `faces` and `corners` separate for stress results.

## Primary comparison

The primary mixture weights are 0.75 for mixed settings and 1/12 for each single-drift component. Because the allocation has 144 mixed settings and 16 in each single-drift component, each of the 192 primary settings has equal weight. The stress samples have zero primary weight.

| Procedure | O1: holdings | O2: + beliefs | O6: complete linked | O7: complete unlinked |
|---|---:|---:|---:|---:|
| 1 (`bank=0`) | 1.839 | 1.581 | 0.925 | 0.986 |
| 2 (`bank=1`) | 1.950 | 1.717 | 1.062 | 1.164 |

Values are primary mean interval widths in percentage points, rounded to three decimals. Each cell uses 12,288 evaluation markets. The two rows represent independent fits of the same procedure, not two distinct economic models.

Paired relative width reduction is 100 times the paired mean baseline-minus-expanded width divided by the paired mean baseline width, using jointly nonempty pairs. It is not the mean of market-specific percentage reductions. The same stratified parameter-setting bootstrap draws support the width estimates and paired comparisons. They preserve all within-setting market, information-design, and procedure pairings.

The primary sample had complete pairs, no empty intervals, and 100% observed reference-mean coverage. Three empty intervals occurred in the procedure-1 near-face stress sample. A bootstrap interval degenerate at 0 or 1 records no observed variation; it is not proof that the population probability equals that boundary.

The repository provides machine-readable tables and their reproduction code. It does not contain the manuscript's Word table formatting, prose, or reference library.
