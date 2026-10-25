"""
Loss Given Default, collateral-driven and scenario-sensitive.

For default at time t:

  recoverable_PV_t = CollateralValue x CollateralIndex(Z_t) x (1 - haircut)
                     / (1 + EIR) ^ recovery_years
  shortfall_t      = max(0, 1 - recoverable_PV_t / EAD_t)
  LGD_t            = max(floor, (1 - cure_rate) x shortfall_t x UnsecuredLGD(Z_t))

So the unsecured residual of a secured loan behaves like an unsecured claim,
cures reduce loss, and both collateral values and unsecured recoveries move
with the macro scenario.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COLLATERAL_PARAMS, SEGMENT_PARAMS


def _per_loan(df: pd.DataFrame, table: dict, key: str, attr: str) -> np.ndarray:
    return np.array([getattr(table[v], attr) for v in df[key]], dtype=float)


def lgd_schedule(df: pd.DataFrame, ead: np.ndarray, z_quarterly: np.ndarray) -> np.ndarray:
    """LGD array matching `ead` (n_loans, n_steps) under one scenario's quarterly Z path."""
    n_steps = ead.shape[1]
    z = z_quarterly[:n_steps][None, :]

    haircut = _per_loan(df, COLLATERAL_PARAMS, "collateral_type", "haircut")[:, None]
    rec_years = _per_loan(df, COLLATERAL_PARAMS, "collateral_type", "recovery_years")[:, None]
    c_sens = _per_loan(df, COLLATERAL_PARAMS, "collateral_type", "z_sensitivity")[:, None]
    unsec = _per_loan(df, SEGMENT_PARAMS, "segment", "unsecured_lgd")[:, None]
    cure = _per_loan(df, SEGMENT_PARAMS, "segment", "cure_rate")[:, None]
    floor = _per_loan(df, SEGMENT_PARAMS, "segment", "lgd_floor")[:, None]
    u_sens = _per_loan(df, SEGMENT_PARAMS, "segment", "lgd_z_sensitivity")[:, None]

    cval = df["collateral_value"].to_numpy(dtype=float)[:, None]
    eir = df["eir"].to_numpy(dtype=float)[:, None]

    coll_index = np.clip(1 + c_sens * z, 0.6, 1.25)
    recoverable_pv = cval * coll_index * (1 - haircut) / (1 + eir) ** rec_years
    with np.errstate(divide="ignore", invalid="ignore"):
        shortfall = np.where(ead > 0, np.clip(1 - recoverable_pv / ead, 0, 1), 0.0)
    unsec_z = np.clip(unsec * (1 - u_sens * z), 0, 1)
    lgd = np.maximum(floor, (1 - cure) * shortfall * unsec_z)
    return np.where(ead > 0, lgd, 0.0)
