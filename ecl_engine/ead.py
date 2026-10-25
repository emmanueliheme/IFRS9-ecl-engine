"""
Exposure at Default on a quarterly grid.

- Amortising: annuity schedule at the EIR with optional constant prepayment.
- Bullet: drawn balance until maturity.
- Revolving: drawn + CCF x undrawn, held flat over the behavioural life
  (IFRS 9 5.5.20: the exposure period for revolvers is the period over
  which the bank is exposed, not the contractual notice period).

Exposure for default in quarter q is the balance at the start of q.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PRODUCTS


def exposure_life_years(df: pd.DataFrame) -> np.ndarray:
    life = df["remaining_term_years"].to_numpy(dtype=float).copy()
    for prod, pp in PRODUCTS.items():
        if pp.facility_type == "revolving":
            m = (df["product"] == prod).to_numpy()
            life[m] = pp.behavioural_life_years
    # unknown revolving products: default to 2 years
    rev_unknown = (df["facility_type"] == "revolving").to_numpy() & ~df["product"].isin(PRODUCTS).to_numpy()
    life[rev_unknown] = 2.0
    return np.maximum(life, 0.25)


def n_quarters(life_years: np.ndarray, steps_per_year: int = 4) -> np.ndarray:
    return np.ceil(np.round(life_years * steps_per_year, 6)).astype(int)


def ead_schedule(df: pd.DataFrame, n_steps: int, steps_per_year: int = 4) -> np.ndarray:
    """Return EAD array (n_loans, n_steps); zero after the exposure life."""
    life = exposure_life_years(df)
    nq = n_quarters(life, steps_per_year)
    k = np.arange(n_steps)[None, :]
    alive = k < nq[:, None]

    drawn = df["drawn_balance"].to_numpy(dtype=float)[:, None]
    undrawn = df["undrawn_limit"].to_numpy(dtype=float)[:, None]
    ftype = df["facility_type"].to_numpy()
    eir = df["eir"].to_numpy(dtype=float)
    r = ((1 + eir) ** (1 / steps_per_year) - 1)[:, None]
    n = nq[:, None].astype(float)

    # annuity outstanding after k payments
    growth_n = (1 + r) ** n
    amort = drawn * (growth_n - (1 + r) ** k) / np.maximum(growth_n - 1, 1e-12)
    prepay = np.array([PRODUCTS[p].annual_prepayment if p in PRODUCTS else 0.0 for p in df["product"]])
    amort *= (1 - prepay[:, None]) ** (k / steps_per_year)

    ccf = np.array([PRODUCTS[p].ccf if p in PRODUCTS else 0.5 for p in df["product"]])[:, None]
    revolving = drawn + ccf * undrawn
    bullet = np.broadcast_to(drawn, (len(df), n_steps))

    ead = np.where((ftype == "amortising")[:, None], amort,
                   np.where((ftype == "revolving")[:, None], revolving, bullet))
    return np.where(alive, np.maximum(ead, 0.0), 0.0)


def ead_today(df: pd.DataFrame) -> np.ndarray:
    ccf = np.array([PRODUCTS[p].ccf if p in PRODUCTS else 0.5 for p in df["product"]])
    is_rev = (df["facility_type"] == "revolving").to_numpy()
    return df["drawn_balance"].to_numpy(dtype=float) + np.where(is_rev, ccf, 0.0) * df["undrawn_limit"].to_numpy(dtype=float)
