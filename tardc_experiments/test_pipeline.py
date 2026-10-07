from pathlib import Path
import tempfile

import numpy as np

from tardc_pipeline import Config, adjacency, metropolis, simulate, system_model


def test_metropolis_is_doubly_stochastic():
    model = system_model(5, 0)
    w = metropolis(adjacency(5, model["edges"]))
    assert np.allclose(w.sum(axis=0), 1.0)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert np.all(w >= 0)


def test_deterministic_run_and_trace():
    cfg = Config(duration=4.0, bootstrap_samples=20)
    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "a.npz"
        p2 = Path(td) / "b.npz"
        a = simulate("P", "S1", 3, cfg, 5, p1)
        b = simulate("P", "S1", 3, cfg, 5, p2)
        assert a["frequency_rmse_hz"] == b["frequency_rmse_hz"]
        assert p1.exists() and p2.exists()


def test_all_controller_scenario_combinations_are_finite():
    cfg = Config(duration=2.0, bootstrap_samples=20)
    for controller in ["B0", "B1", "B2", "B3", "B4", "B5", "P"]:
        for scenario in [f"S{i}" for i in range(8)]:
            metrics = simulate(controller, scenario, 0, cfg)
            numeric = [v for v in metrics.values() if isinstance(v, (float, int))]
            assert all(np.isfinite(numeric))


if __name__ == "__main__":
    tests = [test_metropolis_is_doubly_stochastic,
             test_deterministic_run_and_trace,
             test_all_controller_scenario_combinations_are_finite]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")
