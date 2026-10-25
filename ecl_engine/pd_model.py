"""
Lifetime PD term structures.

1. A through-the-cycle (TTC) annual rating migration matrix per segment
   (8 grades + absorbing default) is either built from assumptions or
   estimated from rating snapshots with the cohort method.
2. Matrix powers give cumulative TTC default probabilities by grade and
   horizon; these capture rating migration over a facility's life.
3. Conditional annual hazards are converted to point-in-time (PIT) with a
   Vasicek-style Z-shift on the probit scale, using the scenario Z per year:

       h_PIT = Phi( Phi^-1(h_TTC) - Z * sqrt(rho / (1 - rho)) )

   The slope sqrt(rho/(1-rho)) is the Vasicek sensitivity of the default
   threshold to the systematic factor. Unlike the textbook conditional-PD
   formula, this form returns exactly h_TTC when Z = 0, so the neutral
   (long-run) state reproduces the TTC term structure.

4. Annual hazards are split to the quarterly grid used by the ECL engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import DEFAULT_STATE, N_GRADES, SEGMENT_PARAMS, STAY_PROB, TTC_ANNUAL_PD

_EPS = 1e-9


def build_transition_matrix(segment: str) -> np.ndarray:
    """Assumption-based annual migration matrix, shape (9, 9), rows sum to 1."""
    mult = SEGMENT_PARAMS[segment].pd_multiplier
    k = N_GRADES + 1
    m = np.zeros((k, k))
    for i in range(N_GRADES):
        pd_i = min(TTC_ANNUAL_PD[i] * mult, 0.5)
        stay = min(STAY_PROB[i], 1 - pd_i)
        move = 1 - pd_i - stay
        up_idx = list(range(i - 1, -1, -1))           # nearest better grade first
        down_idx = list(range(i + 1, N_GRADES))       # nearest worse grade first
        down_share = 0.6 if up_idx and down_idx else (1.0 if down_idx else 0.0)
        for idx_list, share in ((down_idx, down_share), (up_idx, 1 - down_share)):
            if not idx_list or share == 0:
                continue
            w = 0.35 ** np.arange(len(idx_list))       # geometric decay with distance
            w = w / w.sum() * move * share
            for j, wj in zip(idx_list, w):
                m[i, j] += wj
        m[i, i] += stay
        m[i, DEFAULT_STATE] += pd_i
    m[DEFAULT_STATE, DEFAULT_STATE] = 1.0
    return m


def estimate_transition_matrix(from_grade: pd.Series, to_grade: pd.Series) -> np.ndarray:
    """
    Cohort-method estimate from two annual snapshots.
    Grades are 1..8; use 9 for default in `to_grade`.
    Rows with no observations fall back to staying in grade.
    """
    k = N_GRADES + 1
    counts = np.zeros((k, k))
    for f, t in zip(from_grade.astype(int) - 1, to_grade.astype(int) - 1):
        counts[f, t] += 1
    m = np.zeros((k, k))
    for i in range(k):
        n = counts[i].sum()
        m[i] = counts[i] / n if n > 0 else np.eye(k)[i]
    m[DEFAULT_STATE] = np.eye(k)[DEFAULT_STATE]
    return m


def cumulative_pd(matrix: np.ndarray, max_years: int) -> np.ndarray:
    """Cumulative TTC PD, shape (8 grades, max_years + 1); column t = horizon t years."""
    out = np.zeros((N_GRADES, max_years + 1))
    p = np.eye(matrix.shape[0])
    for t in range(1, max_years + 1):
        p = p @ matrix
        out[:, t] = p[:N_GRADES, DEFAULT_STATE]
    return out


def annual_hazards(cpd: np.ndarray) -> np.ndarray:
    """Conditional annual default probabilities h_t = (CPD_t - CPD_{t-1}) / (1 - CPD_{t-1})."""
    surv_prev = 1 - cpd[:, :-1]
    return np.clip((cpd[:, 1:] - cpd[:, :-1]) / np.maximum(surv_prev, _EPS), 0, 1)


def pit_hazards(ttc_hazards: np.ndarray, z: np.ndarray, rho: float) -> np.ndarray:
    """Z-shift of TTC hazards (grades x years) on the annual Z path; Z = 0 returns TTC."""
    h = np.clip(ttc_hazards, _EPS, 1 - _EPS)
    zz = z[: h.shape[1]][None, :]
    return norm.cdf(norm.ppf(h) - zz * np.sqrt(rho / (1 - rho)))


def quarterly_hazards(annual: np.ndarray, steps_per_year: int = 4) -> np.ndarray:
    """Split each annual hazard evenly in survival terms: 1 - (1 - h)^(1/n)."""
    q = 1 - (1 - annual) ** (1 / steps_per_year)
    return np.repeat(q, steps_per_year, axis=1)


def interp_cpd(cpd_row_matrix: np.ndarray, grade_idx: np.ndarray, horizon_years: np.ndarray) -> np.ndarray:
    """Linear interpolation of cumulative PD for each (grade, fractional horizon)."""
    max_t = cpd_row_matrix.shape[1] - 1
    h = np.clip(horizon_years, 0, max_t)
    lo = np.floor(h).astype(int)
    hi = np.minimum(lo + 1, max_t)
    frac = h - lo
    a = cpd_row_matrix[grade_idx, lo]
    b = cpd_row_matrix[grade_idx, hi]
    return a + (b - a) * frac


def pd_term_structure_table(segment: str, max_years: int = 10, z: np.ndarray | None = None) -> pd.DataFrame:
    """Cumulative PD by grade for charts; optionally conditioned on a Z path."""
    from .config import GRADES
    m = build_transition_matrix(segment)
    cpd = cumulative_pd(m, max_years)
    if z is not None:
        hz = pit_hazards(annual_hazards(cpd), z, SEGMENT_PARAMS[segment].asset_correlation)
        cpd = np.hstack([np.zeros((N_GRADES, 1)), 1 - np.cumprod(1 - hz, axis=1)])
    df = pd.DataFrame(cpd, index=GRADES, columns=list(range(max_years + 1)))
    df.index.name = "grade"
    return df
