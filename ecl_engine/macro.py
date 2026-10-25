"""
Forward-looking information: macro scenarios -> systematic credit factor Z.

A linear satellite model maps standardised macro variables to a single
credit-cycle index Z (Z > 0 benign, Z < 0 stressed). Explicit forecasts
cover years 1-3; Z then reverts linearly to 0 (the through-the-cycle
state) over MEAN_REVERSION_YEARS, as is common practice for horizons
beyond reasonable and supportable forecasts (IFRS 9 B5.5.54).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import MACRO_VARIABLES, MEAN_REVERSION_YEARS


def z_from_macro(values: Dict[str, float]) -> float:
    z = 0.0
    for name, (mean, std, beta) in MACRO_VARIABLES.items():
        z += beta * (values[name] - mean) / std
    return z


def z_path(scenario: dict, max_years: int) -> np.ndarray:
    """Annual Z for years 1..max_years (index 0 = first 12 months)."""
    n_explicit = len(scenario["gdp_growth"])
    explicit = np.array([
        z_from_macro({k: scenario[k][t] for k in MACRO_VARIABLES}) for t in range(n_explicit)
    ])
    path = np.zeros(max_years)
    path[:n_explicit] = explicit[:max_years]
    last = explicit[-1]
    for j in range(MEAN_REVERSION_YEARS):
        idx = n_explicit + j
        if idx < max_years:
            path[idx] = last * (1 - (j + 1) / (MEAN_REVERSION_YEARS + 1))
    return path


def scenario_table(scenarios: dict, weights: Dict[str, float], max_years: int = 6) -> pd.DataFrame:
    rows = []
    for name, sc in scenarios.items():
        zp = z_path(sc, max_years)
        for t in range(len(sc["gdp_growth"])):
            rows.append({"scenario": name, "weight": weights[name], "year": t + 1,
                         **{k: sc[k][t] for k in MACRO_VARIABLES}, "Z": round(zp[t], 3)})
    return pd.DataFrame(rows)
