# Changes

## Package 1.0.0 — October 15, 2026

Prepared replication materials for manuscript version 2. The economic model, inference routines, calibration, seeds, fitted models, and recorded Monte Carlo results are unchanged.

- Preserved 49 archived source, computational summary, and test files under `engine/`.
- Added a separate command-line entry point, pinned dependencies, checksum verification, isolated work directories, citation metadata, and documentation.
- Split the retained numerical records into main-experiment and calibration/development Release attachments. Added hashes and archive provenance.
- Added reproducible figure layouts using the paper's study design and numerical summary tables.
- Excluded manuscript drafts, manuscript-generation scripts, document-verification records, and redundant historical plot exports from the numerical attachments.
- Verified original tests, selected exact simulation/feature/interval replays, regenerated summaries, and refits from saved training/calibration draws. See `provenance/package-validation.json` for the scope of these checks.

## Experiment 0.6.0 — completed October 13, 2026

The main experiment evaluated the frozen 0.5.0 procedures at 256 newly drawn settings. It retained 131,072 intervals, 262,144 independent reference markets, and 18,432 endpoint-audit markets. Main-run results are distinct from the earlier calibration and preliminary validation.

## Calibration 0.5.0 and earlier development

The calibration attachment retains final cutoff calibration, preliminary validation, and the earlier expanded and repaired training banks. Original stage-specific protocols appear in `docs/history/`. Statements in those protocols about a future confirmation run refer to their original dates.
