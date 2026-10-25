import numpy as np
import pandas as pd
import pytest

from ecl_engine import EngineConfig, StagingPolicy, generate_portfolio, run_ecl
from ecl_engine.config import N_GRADES, SEGMENTS
from ecl_engine.ead import ead_schedule
from ecl_engine.macro import z_path
from ecl_engine.pd_model import (annual_hazards, build_transition_matrix, cumulative_pd,
                                 estimate_transition_matrix, pit_hazards)
from ecl_engine.staging import assign_stages


@pytest.fixture(scope="module")
def result():
    return run_ecl(generate_portfolio(1500, seed=7))


# ---------------- PD ----------------
@pytest.mark.parametrize("seg", SEGMENTS)
def test_transition_matrix_is_stochastic(seg):
    m = build_transition_matrix(seg)
    assert np.allclose(m.sum(axis=1), 1.0)
    assert (m >= 0).all()
    assert m[-1, -1] == 1.0


@pytest.mark.parametrize("seg", SEGMENTS)
def test_cumulative_pd_monotone_in_time_and_grade(seg):
    cpd = cumulative_pd(build_transition_matrix(seg), 15)
    assert (np.diff(cpd, axis=1) >= -1e-12).all()       # non-decreasing over time
    assert (np.diff(cpd[:, 1], axis=0) > 0).all()       # worse grade -> higher 1y PD


def test_vasicek_direction():
    h = annual_hazards(cumulative_pd(build_transition_matrix("SME"), 5))
    bad = pit_hazards(h, np.full(5, -1.0), 0.12)
    good = pit_hazards(h, np.full(5, 1.0), 0.12)
    neutral = pit_hazards(h, np.zeros(5), 0.12)
    assert (bad > neutral).all() and (neutral > good).all()
    assert np.allclose(neutral, np.clip(h, 1e-9, 1 - 1e-9), atol=1e-8)


def test_cohort_estimator():
    est = estimate_transition_matrix(pd.Series([1, 1, 1, 2]), pd.Series([1, 2, 9, 2]))
    assert np.isclose(est[0, 0], 1 / 3) and np.isclose(est[0, N_GRADES], 1 / 3)
    assert np.allclose(est.sum(axis=1), 1.0)


def test_z_path_reverts_to_ttc():
    sc = {"gdp_growth": [1, 1, 1], "inflation": [30, 30, 30], "oil_price": [50, 50, 50], "policy_rate": [30, 30, 30]}
    zp = z_path(sc, 10)
    assert zp[0] < 0 and zp[-1] == 0 and abs(zp[3]) < abs(zp[2])


# ---------------- EAD ----------------
def test_amortising_schedule_runs_down():
    df = pd.DataFrame([dict(product="SME Term Loan", facility_type="amortising", remaining_term_years=2.0,
                            eir=0.2, drawn_balance=1000.0, undrawn_limit=0.0)])
    e = ead_schedule(df, 12)[0]
    assert e[0] == pytest.approx(1000.0)
    assert (np.diff(e[:8]) < 0).all() and (e[8:] == 0).all()


def test_revolving_uses_ccf_and_behavioural_life():
    df = pd.DataFrame([dict(product="Credit Card", facility_type="revolving", remaining_term_years=1.0,
                            eir=0.3, drawn_balance=600.0, undrawn_limit=400.0)])
    e = ead_schedule(df, 20)[0]
    assert e[0] == pytest.approx(600 + 0.65 * 400)
    assert (e[:12] > 0).all() and (e[12:] == 0).all()   # 3-year behavioural life


# ---------------- Staging ----------------
def _one(**kw):
    base = dict(dpd=0, default_flag=0, watchlist=0, forborne=0, current_grade=4, orig_grade=4)
    base.update(kw)
    return pd.DataFrame([base])


@pytest.mark.parametrize("kw,expected,trigger", [
    (dict(dpd=95), 3, "DPD_90"),
    (dict(default_flag=1), 3, "DEFAULT_FLAG"),
    (dict(dpd=35), 2, "DPD_30"),
    (dict(watchlist=1), 2, "WATCHLIST"),
    (dict(forborne=1), 2, "FORBEARANCE"),
    (dict(current_grade=7), 2, "DOWNGRADE"),
    (dict(), 1, ""),
])
def test_staging_rules(kw, expected, trigger):
    out = assign_stages(_one(**kw), np.array([0.05]), np.array([0.02]), np.array([0.05]), StagingPolicy())
    assert out["stage"].iloc[0] == expected
    assert trigger in out["sicr_triggers"].iloc[0]


def test_quantitative_sicr_needs_relative_and_absolute():
    p = StagingPolicy()
    rel_only = assign_stages(_one(), np.array([0.008]), np.array([0.004]), np.array([0.003]), p)
    both = assign_stages(_one(), np.array([0.10]), np.array([0.04]), np.array([0.04]), p)
    assert rel_only["stage"].iloc[0] == 1 and both["stage"].iloc[0] == 2


def test_low_credit_risk_exemption_blocks_quantitative_only():
    p = StagingPolicy(low_credit_risk_exemption=True)
    lcr = assign_stages(_one(current_grade=7, orig_grade=1), np.array([0.05]), np.array([0.001]), np.array([0.01]), p)
    assert lcr["stage"].iloc[0] == 1
    lcr_dpd = assign_stages(_one(dpd=40), np.array([0.05]), np.array([0.001]), np.array([0.01]), p)
    assert lcr_dpd["stage"].iloc[0] == 2


# ---------------- ECL ----------------
def test_ecl_ordering_and_weighting(result):
    l, s = result.loans, result.scenario_ecl
    assert (l["ecl"] >= 0).all()
    assert (l["ecl_lifetime"] + 1e-6 >= l["ecl_12m"]).all()
    w = np.array([result.weights[c] for c in s.columns])
    assert np.allclose(s.to_numpy() @ w, l["ecl"].to_numpy())
    tot = s.sum()
    assert tot["Downside"] > tot["Base"] > tot["Upside"]


def test_stage_measurement_basis(result):
    l = result.loans
    s1, s2, s3 = (l[l.stage == k] for k in (1, 2, 3))
    assert np.allclose(s1["ecl"], s1["ecl_12m"])
    assert np.allclose(s2["ecl"], s2["ecl_lifetime"])
    assert (s3["pd_12m"] == 1).all()
    cov = [g["ecl"].sum() / g["ead_today"].sum() for g in (s1, s2, s3)]
    assert cov[0] < cov[1] < cov[2]


def test_weights_normalised():
    cfg = EngineConfig()
    for sc in cfg.scenarios.values():
        sc["weight"] *= 3
    assert sum(cfg.normalised_weights().values()) == pytest.approx(1.0)
