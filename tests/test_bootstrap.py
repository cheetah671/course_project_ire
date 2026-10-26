import numpy as np

from ragroute.stats.bootstrap import best_config_stability, holm, paired_bootstrap


def test_identical_systems_ci_contains_zero():
    rng = np.random.default_rng(0)
    a = rng.uniform(size=500)
    r = paired_bootstrap(a, a.copy(), n_resamples=2000)
    assert r.diff == 0 and r.ci_low <= 0 <= r.ci_high and r.p_value > 0.9


def test_clear_difference_detected():
    rng = np.random.default_rng(1)
    b = (rng.uniform(size=800) < 0.5).astype(float)
    a = np.clip(b + (rng.uniform(size=800) < 0.15), 0, 1)
    r = paired_bootstrap(a, b, n_resamples=5000)
    assert r.ci_low > 0 and r.p_value < 0.01
    assert abs(r.diff - (a.mean() - b.mean())) < 1e-12


def test_stratified_macro_weights_strata_equally():
    a = np.r_[np.ones(900), np.zeros(100)]
    b = np.zeros(1000)
    strata = np.r_[np.zeros(900), np.ones(100)]
    micro = paired_bootstrap(a, b, n_resamples=500, strata=strata, statistic="micro")
    macro = paired_bootstrap(a, b, n_resamples=500, strata=strata, statistic="macro")
    assert abs(micro.diff - 0.9) < 1e-12 and abs(macro.diff - 0.5) < 1e-12


def test_holm_monotone_and_bounded():
    adj = holm([0.01, 0.04, 0.03, 0.5])
    assert np.all(adj <= 1) and adj[0] == 0.04 and adj[3] == 0.5


def test_best_config_stability_flags_ties():
    rng = np.random.default_rng(2)
    Q = (rng.uniform(size=(300, 3)) < [0.5, 0.5, 0.2]).astype(float)
    r = best_config_stability(Q, ["a", "b", "c"], n_resamples=1000)
    assert r["best"] in ("a", "b") and not r["claim_supported"]
