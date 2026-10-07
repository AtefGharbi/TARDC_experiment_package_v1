# TARDC 30-seed experiment summary

The frozen experiment produced 1,950 primary runs and 1,650 paired
counterfactual runs. The latter retain the same controller, topology, physical
events, and seed while disabling the cyber attack.

## Main findings

- Across the claimed S0-S6 scope, TARDC completed all 210 System A trials
  within the declared 49.5-50.5 Hz and 380-420 V safety envelope.
- TARDC's mean frequency RMSE across S0-S6 was 0.18075 Hz. The paired
  difference from fixed-rate TARDC was -0.000002 Hz (Holm-adjusted p=0.839).
- Adaptive triggering reduced mean traffic from 1,759.8 to 606.4 packets per
  trial relative to fixed-rate TARDC, a 65.5% reduction.
- Across S1-S6, TARDC's mean maximum attack-caused frequency difference from
  the matched counterfactual was 0.01135 Hz. The corresponding value for
  ordinary dynamic consensus was 0.27056 Hz.
- Fixed-rate TARDC retained a small resilience advantage: its paired maximum
  attack-caused difference was 0.00292 Hz lower on average.
- No normal link was falsely isolated by TARDC in S0-S6.
- S7 violates the F-local assumption. Its TARDC safe-trial rate was zero, with
  a mean maximum attack-caused frequency difference of 1.37846 Hz.
- Under S0, the measured mean Python update time increased from 0.431 ms for
  five agents to approximately 2.1 ms for 50 agents. These are software timing
  measurements, not real-time hardware benchmarks.

## Interpretation boundary

The electrical layer is an aggregate reduced-order dynamic benchmark
parameterized from the published five-generator system. It is not the original
PSCAD/EMTDC model, which was not released, and it does not support EMT-level,
hardware-level, or nonlinear-stability claims.

All detailed values are in `run_metrics.csv`, `aggregate_metrics.csv`, and
`statistical_tests.csv`. The frozen settings and hashes are in `manifest.json`.
