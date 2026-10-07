# TARDC reproducible experiment pipeline

This package implements the experiments prespecified in Sections 6 and 7 of
the TARDC manuscript. It uses the five-generator ratings, droop coefficients,
loads, graph, 0.2 s coordination interval, and switching sequence reported by
Li et al. (IEEE/CAA Journal of Automatica Sinica, 2016).

## Scientific scope

The source article implemented its electrical system in PSCAD/EMTDC but did
not release the project or every network/controller parameter. This package is
therefore a **reduced-order dynamic benchmark**, not an electromagnetic-
transient replication. It is suitable for testing the relative behavior of
consensus, trust, filtering, attack, and communication mechanisms. Claims of
EMT-level validation require a later PSCAD, Simulink/Simscape, or OPAL-RT study.

## Run

Use Python 3.11+ with NumPy, pandas, SciPy, and Matplotlib.

```bash
python tardc_pipeline.py --output results/full --seeds 30
```

For a short functional check:

```bash
python tardc_pipeline.py --output results/quick --seeds 2 --quick
```

Run the dependency-light validation tests with:

```bash
python test_pipeline.py
```

## Outputs

- `manifest.json`: frozen configuration, versions, seeds, and configuration hash.
- `calibration.json`: disjoint development calibration record.
- `traces/*.npz`: compressed time series for every run.
- `counterfactual_traces/*.npz`: matched traces with the cyber attack disabled.
- `run_metrics.csv`: one row per controller, scenario, seed, and network size.
- `counterfactual_metrics.csv`: metrics for the matched no-attack controls.
- `aggregate_metrics.csv`: mean, standard deviation, median, and bootstrap CI.
- `statistical_tests.csv`: Friedman and Holm-corrected Wilcoxon tests.
- `figures/figure3_*.png` through `figure8_*.png`: prespecified manuscript figures.

Evaluation seeds are paired across controllers. Scenario S7 deliberately
violates the F-local assumption and must be described as a failure-boundary
stress test rather than evidence of guaranteed tolerance.
