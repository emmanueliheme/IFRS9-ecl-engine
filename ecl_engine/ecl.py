"""
ECL orchestration.

For each scenario s and quarterly step q (default in (t_q, t_q + 1/4]):

    ECL_s = sum_q  S_s(t_q) * h_s(q)  x  LGD_s(q)  x  EAD(q)  x  (1 + EIR)^-(t_q + 1/4)
            '---- marginal PD ----'

    Stage 1 : sum over the first 12 months only
    Stage 2 : sum over the exposure life (lifetime ECL)
    Stage 3 : PD = 1, ECL = EAD_today x LGD_s(0)

Probability-weighted ECL = sum_s w_s x ECL_s  (IFRS 9 5.5.17(a)).
Staging is performed once, on scenario-weighted PDs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .config import N_GRADES, SEGMENT_PARAMS, SEGMENTS, EngineConfig
from .data import validate_portfolio
from .ead import ead_schedule, ead_today, exposure_life_years, n_quarters
from .lgd import lgd_schedule
from .macro import z_path
from .pd_model import (annual_hazards, build_transition_matrix, cumulative_pd, interp_cpd,
                       pit_hazards, quarterly_hazards)
from .staging import assign_stages


@dataclass
class ECLResult:
    loans: pd.DataFrame                    # loan-level results
    scenario_ecl: pd.DataFrame             # loan x scenario final ECL (under final stage)
    weights: Dict[str, float]
    z_paths: Dict[str, np.ndarray]
    config: EngineConfig


def _marginal_pd(df: pd.DataFrame, z_annual: np.ndarray, cfg: EngineConfig,
                 grade_col: str, n_steps: int, life_q: np.ndarray) -> np.ndarray:
    """Marginal (unconditional) PD per loan per quarter, zero beyond exposure life."""
    out = np.zeros((len(df), n_steps))
    for seg in SEGMENTS:
        m = (df["segment"] == seg).to_numpy()
        if not m.any():
            continue
        cpd = cumulative_pd(build_transition_matrix(seg), cfg.max_years)
        h_pit = pit_hazards(annual_hazards(cpd), z_annual, SEGMENT_PARAMS[seg].asset_correlation)
        hq = quarterly_hazards(h_pit, cfg.steps_per_year)[:, :n_steps]       # grades x steps
        g = df.loc[m, grade_col].to_numpy().astype(int) - 1
        h_loan = hq[g]
        surv_prev = np.cumprod(np.hstack([np.ones((m.sum(), 1)), 1 - h_loan[:, :-1]]), axis=1)
        out[m] = surv_prev * h_loan
    alive = np.arange(n_steps)[None, :] < life_q[:, None]
    return np.where(alive, out, 0.0)


def _origination_lifetime_pd(df: pd.DataFrame, cfg: EngineConfig, life_years: np.ndarray) -> np.ndarray:
    """
    PD that was expected at origination for the window [age, age + remaining life],
    using the TTC term structure of the origination grade (forward PD).
    """
    out = np.zeros(len(df))
    age = df["age_years"].to_numpy(dtype=float)
    for seg in SEGMENTS:
        m = (df["segment"] == seg).to_numpy()
        if not m.any():
            continue
        cpd = cumulative_pd(build_transition_matrix(seg), cfg.max_years + 10)
        g = df.loc[m, "orig_grade"].to_numpy().astype(int) - 1
        c0 = interp_cpd(cpd, g, age[m])
        c1 = interp_cpd(cpd, g, age[m] + life_years[m])
        out[m] = (c1 - c0) / np.maximum(1 - c0, 1e-9)
    return out


def run_ecl(df: pd.DataFrame, cfg: EngineConfig | None = None) -> ECLResult:
    cfg = cfg or EngineConfig()
    validate_portfolio(df)
    df = df.reset_index(drop=True).copy()
    spy = cfg.steps_per_year
    n_steps = cfg.max_years * spy
    weights = cfg.normalised_weights()

    life_y = np.minimum(exposure_life_years(df), cfg.max_years)
    life_q = n_quarters(life_y, spy)
    q12 = min(spy, n_steps)

    ead = ead_schedule(df, n_steps, spy)
    t_end = (np.arange(n_steps) + 1) / spy
    eir = df["eir"].to_numpy(dtype=float)[:, None]
    df_factor = (1 + eir) ** (-t_end[None, :])
    ead0 = ead_today(df)

    z_paths, per_s = {}, {}
    lpd_w = np.zeros(len(df)); pd12_w = np.zeros(len(df))
    lpd_prior_w = np.zeros(len(df)); pd12_prior_w = np.zeros(len(df))
    for name, sc in cfg.scenarios.items():
        za = z_path(sc, cfg.max_years)
        zq = np.repeat(za, spy)
        z_paths[name] = za
        mpd = _marginal_pd(df, za, cfg, "current_grade", n_steps, life_q)
        lgd = lgd_schedule(df, ead, zq)
        loss = mpd * lgd * ead * df_factor
        lgd0 = lgd_schedule(df, ead0[:, None], zq[:1])[:, 0]
        per_s[name] = {
            "ecl_12m": loss[:, :q12].sum(axis=1),
            "ecl_lifetime": loss.sum(axis=1),
            "ecl_stage3": ead0 * lgd0,
            "pd_12m": mpd[:, :q12].sum(axis=1),
            "pd_lifetime": mpd.sum(axis=1),
            "lgd_0": lgd0,
        }
        w = weights[name]
        lpd_w += w * per_s[name]["pd_lifetime"]
        pd12_w += w * per_s[name]["pd_12m"]
        # prior-period PDs for the migration matrix (prior grade, one quarter more life)
        mpd_prior = _marginal_pd(df, za, cfg, "prior_grade", n_steps, np.minimum(life_q + 1, n_steps))
        lpd_prior_w += w * mpd_prior.sum(axis=1)
        pd12_prior_w += w * mpd_prior[:, :q12].sum(axis=1)

    lpd_orig = _origination_lifetime_pd(df, cfg, life_y)
    stages = assign_stages(df, lpd_w, pd12_w, lpd_orig, cfg.staging)
    lpd_orig_prior = _origination_lifetime_pd(df, cfg, life_y + 1 / spy)
    prior = assign_stages(df, lpd_prior_w, pd12_prior_w, lpd_orig_prior, cfg.staging,
                          grade_col="prior_grade", dpd_col="prior_dpd", use_flags=False)

    stage = stages["stage"].to_numpy()
    scen_ecl = {}
    for name, r in per_s.items():
        scen_ecl[name] = np.select([stage == 1, stage == 2, stage == 3],
                                   [r["ecl_12m"], r["ecl_lifetime"], r["ecl_stage3"]])
    scen_df = pd.DataFrame(scen_ecl)
    w_vec = np.array([weights[c] for c in scen_df.columns])
    ecl = scen_df.to_numpy() @ w_vec

    ecl12_w = sum(weights[s] * per_s[s]["ecl_12m"] for s in per_s)
    ecllt_w = sum(weights[s] * per_s[s]["ecl_lifetime"] for s in per_s)
    lgd0_w = sum(weights[s] * per_s[s]["lgd_0"] for s in per_s)

    loans = df.copy()
    loans["exposure_life_years"] = life_y
    loans["ead_today"] = ead0
    loans["pd_12m"] = np.where(stage == 3, 1.0, pd12_w)
    loans["pd_lifetime"] = np.where(stage == 3, 1.0, lpd_w)
    loans["pd_lifetime_at_origination"] = lpd_orig
    loans["lgd_today"] = lgd0_w
    loans = pd.concat([loans, stages], axis=1)
    loans["prior_stage"] = prior["stage"].to_numpy()
    loans["ecl_12m"] = ecl12_w
    loans["ecl_lifetime"] = np.where(stage == 3, ecl, ecllt_w)
    loans["ecl"] = ecl
    loans["coverage"] = np.where(ead0 > 0, ecl / ead0, 0.0)
    scen_df.index = loans["loan_id"]
    return ECLResult(loans=loans, scenario_ecl=scen_df, weights=weights, z_paths=z_paths, config=cfg)
