# Reproduction guide

## Environment and files

Use Linux with Python 3.12 and `g++`. The packaging checks used Python 3.12.14 and the exact direct dependency versions in `requirements.txt`. Native Windows is not a supported execution target for the original compiled kernel: use WSL2. Other platforms and numerical libraries have not been checked. Run commands from the repository root.

The release has two data attachments:

| Attachment | Compressed size | Contents |
|---|---:|---|
| `monte-carlo-results-v0.6.0.zip` | 100,600,835 bytes | Main evaluation, reference, fit, audit, model, specification, and summary records |
| `calibration-and-development-v0.5.0.zip` | 352,692,768 bytes | Earlier development banks, final calibration, preliminary validation, model objects, and diagnostics |

Put downloaded attachments in `assets/` without extracting them. Alternatively, pass `--assets-dir /path/to/attachments`. In the author's upload bundle, the adjacent `release-assets/` directory is detected automatically. Each data ZIP is checked against its recorded size and SHA-256 before extraction. It is extracted only into a route-specific work directory.

Allow at least 5 GB of free space for the supplied verification and refitting routes together. A full run also regenerates raw investor releases; allow extra space and monitor disk use. The `full` route's peak disk requirement has not been benchmarked in this package. Workspaces are separate copies, so running several routes retains duplicate data.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
g++ --version
python reproduce.py check
```

`check` uses only the Python standard library. `figures` needs NumPy, pandas, and Matplotlib. The scientific routes need the full requirements. BLAS/OpenMP worker threads are limited to one per subprocess by the wrapper; the `full` route controls simulation parallelism through `--workers`.

## Rebuild completed main results

```bash
python reproduce.py quick
```

This route copies the frozen source into `work/quick/`, extracts the main attachment, and executes the original `confirmation_summary.py`. It reads all retained evaluation, reference, fitted-interval, and audit records. It rebuilds the seven summary CSVs and the 131,072-row interval table, then draws the figures.

The wrapper compares all 9,116 rows in the seven small summary tables with `reference/`, checks the integrity fields apart from the timestamp, and checks the interval-record count. Numeric summary tolerances are `rtol=1e-10` and `atol=1e-12`. String fields, shapes, and column order must agree. The interval count includes 98,304 primary-sample intervals and 32,768 stress-sample intervals.

Outputs:

- `work/quick/results/confirmation_v06/summary/`
- `work/quick/figures/`

This route takes minutes on the preparation host. It reaggregates completed results; it does not generate new populations or refit the interval endpoints.

## Verify the archived implementation

```bash
python reproduce.py verify
```

Both attachments are required. The route checks the original main-run and calibration source/model freezes, runs all 34 original tests, and executes the original `confirmation_checks.py`. That script checks allocations, drift restrictions, parameter bounds, audit selection, and bootstrap weights. It also replays four archived preliminary markets, their observation summaries, and four procedure-1 O2 interval fits with exact comparisons.

The route then rebuilds the preliminary-validation summaries and the main-run summaries. The main summaries are compared with the archived reference tables. These are implementation checks and selected replays, not a new validation of the model's fit to real-world Bitcoin markets.

## Refit using saved training and calibration draws

```bash
python reproduce.py refit
```

The route fits the original observation functions, response functions, covariance transforms, conditional cutoff functions, and response allowances using the retained expanded/repaired training banks. It then repeats final 0.5.0 cutoff calibration using the 24 retained 512-market calibration files.

Rebuilt models appear under `work/refit/results/reproduction_v04/models/` and `work/refit/results/reproduction_v05/models/`. The wrapper compares all numeric arrays in the eight pre-adjustment and eight final model objects with their archived counterparts (`rtol=1e-8`, `atol=1e-10`), plus the selected model metadata and response allowances.

This route uses the original training and calibration draws. It does not generate new independent training banks or estimate variability across repeated training exercises. It also does not rerun the main Monte Carlo after refitting. The original model copies remain separate.

## Repeat the full main Monte Carlo

```bash
python reproduce.py full --workers 4
```

Only the main attachment is required. The wrapper starts a separate `work/full/` directory containing the frozen source, the main specification, eight fitted models, and the original freeze record. It deliberately omits saved main evaluation, reference, fit, audit, and completion files from that starting workspace.

The original driver then generates the evaluation and independent reference markets, reconstructs the raw analyst releases, fits the response intervals, performs the additional search checks, and generates the independent endpoint audits. It finally rebuilds summaries and compares them with the retained results.

The original seeds reproduce the original experiment. A new seed allocation would be a new experiment. The two trained procedures remain fixed. This route does not rerun the development and calibration sequence.

The main computation previously required roughly four hours in its original environment. Allow several hours and adjust `--workers` to available CPUs and memory. The full driver was not rerun during package preparation. Its starting workspace and frozen inputs were checked; the lower-cost routes were executed.

For an interrupted full run, repeat the same command. The original driver uses retained checkpoints. It does not replace failed simulations with substitute draws. If a numerical comparison differs on another platform, retain the outputs and diagnose the difference before changing tolerances, seeds, or frozen files.

## Figures only

```bash
python reproduce.py figures
```

This reads the small `reference/` tables and writes two figures as PNG, PDF, and SVG to `work/figures/`. Figure 1 shows the experiment design; Figure 2 shows primary mean widths and paired percentage gains. The supplied layouts are reproducible redraws of the working paper's figures; they use the same design and summary values but are not intended to reproduce the Word document's image bytes or typography.

## Work directories and failures

Each route uses its own directory under `work/`. `--workdir /new/path` chooses another location. The wrapper refuses an existing unrelated or incompletely prepared directory. If extraction was interrupted before workspace preparation finished, use a new path or remove only that incomplete generated workspace. Do not remove the downloaded archives or source files.

`check --source-only` detects altered archived source bytes. `.gitattributes` preserves these bytes across Git checkouts. Do not reformat `engine/` or normalize its line endings. Research changes should use a separate branch and a new experiment specification; do not update checksums to make a modified procedure appear to be the frozen baseline.

The original full-run archive omitted raw analyst releases to reduce its size. Saved feature arrays, fitted intervals, and evaluator records are sufficient for `quick`. `full` regenerates the raw releases from their recorded seeds. The omitted releases do not prevent the supplied exact replay checks.

`progress_*.json` records can retain a historical `running` value after their last progress update. Completion is established by the `completed_*.json` markers and the verified summary integrity record, not the last progress string alone.

## Historical development routes

The `engine/pilot/` modules and retained expanded/repaired banks preserve the earlier development sequence. `docs/history/` contains the specifications and protocols as they existed at those stages. They include obsolete forward-looking statements; use the main 0.6.0 specification for the final experiment.

Recreating every exploratory development run is a separate, longer exercise. The tested `refit` route reconstructs the final fitted objects from stored scientific inputs. It is the supplied route for checking the dependency of final calibration on those training inputs.
