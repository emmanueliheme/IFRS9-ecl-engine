"""
Synthetic loan book generator and input schema.

The generator produces a realistic-looking Nigerian bank portfolio (NGN)
across Retail, SME and Corporate segments so the engine can be demonstrated
end to end without confidential data. Real data can be loaded with
`load_portfolio`, which validates it against REQUIRED_COLUMNS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import N_GRADES, PRODUCTS

REQUIRED_COLUMNS = {
    "loan_id": "Unique facility identifier",
    "segment": "Retail | SME | Corporate",
    "product": "Product name (see config.PRODUCTS)",
    "facility_type": "amortising | bullet | revolving",
    "remaining_term_years": "Contractual remaining term (behavioural life is applied to revolvers)",
    "age_years": "Time since origination in years",
    "eir": "Effective interest rate (annual, decimal) used for discounting",
    "drawn_balance": "Gross carrying amount today (NGN)",
    "undrawn_limit": "Undrawn committed amount (NGN), 0 for term loans",
    "collateral_type": "Property | Equipment | Vehicle | Cash | None",
    "collateral_value": "Current market value of collateral (NGN)",
    "orig_grade": "Internal rating at origination, 1 (best) to 8",
    "current_grade": "Internal rating today, 1 to 8",
    "prior_grade": "Internal rating at previous reporting date",
    "dpd": "Days past due today",
    "prior_dpd": "Days past due at previous reporting date",
    "watchlist": "1 if on credit watchlist",
    "forborne": "1 if forbearance / restructuring granted",
    "default_flag": "1 if credit-impaired on qualitative grounds (unlikely-to-pay), independent of DPD",
}

_GRADE_DIST = {
    "Retail":    [0.02, 0.06, 0.16, 0.26, 0.24, 0.15, 0.08, 0.03],
    "SME":       [0.01, 0.04, 0.12, 0.24, 0.26, 0.19, 0.10, 0.04],
    "Corporate": [0.06, 0.14, 0.26, 0.26, 0.16, 0.08, 0.03, 0.01],
}


def _dpd_from_grade(rng: np.random.Generator, grade: np.ndarray) -> np.ndarray:
    """DPD buckets whose delinquency probability rises with worse grades."""
    g = grade / (N_GRADES - 1)  # 0 (best) .. 1 (worst)
    p_late = 0.02 + 0.30 * g ** 2
    bucket_probs = np.array([0.55, 0.22, 0.13, 0.10])  # 1-29, 30-59, 60-89, 90+
    late = rng.random(grade.size) < p_late
    bucket = rng.choice(4, size=grade.size, p=bucket_probs)
    lo = np.array([1, 30, 60, 90])[bucket]
    hi = np.array([29, 59, 89, 360])[bucket]
    dpd = rng.integers(lo, hi + 1)
    return np.where(late, dpd, 0)


def generate_portfolio(n_loans: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(PRODUCTS)
    mix = np.array([PRODUCTS[p].mix for p in names])
    product = rng.choice(names, size=n_loans, p=mix / mix.sum())

    rows = []
    for i, prod in enumerate(product):
        pp = PRODUCTS[prod]
        term = float(rng.integers(pp.term_years[0], pp.term_years[1] + 1))
        if pp.facility_type == "revolving":
            age = float(rng.uniform(0.25, 6.0))
            remaining = 1.0  # contractual annual renewal; behavioural life applied in EAD/PD
        else:
            age = float(np.round(rng.uniform(0.0, term - 0.25), 2))
            remaining = float(np.round(term - age, 2))
        amount = float(pp.amount_median_ngn * rng.lognormal(0.0, 0.6))
        if pp.facility_type == "revolving":
            utilisation = float(rng.beta(2.5, 2.0))
            drawn, undrawn = amount * utilisation, amount * (1 - utilisation)
        elif pp.facility_type == "amortising":
            drawn, undrawn = amount * max(remaining / term, 0.05), 0.0
        else:
            drawn, undrawn = amount, 0.0
        ctypes = list(pp.collateral_mix)
        cprob = np.array([pp.collateral_mix[c] for c in ctypes])
        ctype = str(rng.choice(ctypes, p=cprob))
        cval = 0.0 if ctype == "None" else amount / rng.uniform(*pp.ltv_range)
        rows.append(dict(
            loan_id=f"LN{i + 1:06d}", segment=pp.segment, product=prod,
            facility_type=pp.facility_type, remaining_term_years=remaining, age_years=age,
            eir=float(np.round(rng.uniform(*pp.eir_range), 4)),
            drawn_balance=round(drawn, 2), undrawn_limit=round(undrawn, 2),
            collateral_type=ctype, collateral_value=round(cval, 2),
        ))
    df = pd.DataFrame(rows)

    # Ratings: origination grade, migration since origination, last-quarter move
    orig = np.empty(n_loans, dtype=int)
    for seg, dist in _GRADE_DIST.items():
        m = (df["segment"] == seg).to_numpy()
        orig[m] = rng.choice(N_GRADES, size=m.sum(), p=dist)
    drift = rng.choice([-1, 0, 1, 2, 3, 4], size=n_loans, p=[0.12, 0.58, 0.16, 0.08, 0.04, 0.02])
    prior = np.clip(orig + drift, 0, N_GRADES - 1)
    q_move = rng.choice([-1, 0, 1, 2], size=n_loans, p=[0.05, 0.85, 0.08, 0.02])
    current = np.clip(prior + q_move, 0, N_GRADES - 1)

    dpd = _dpd_from_grade(rng, current)
    # prior DPD: one bucket earlier for most delinquent loans (roll-rate behaviour)
    prior_dpd = np.where(rng.random(n_loans) < 0.6, np.maximum(dpd - 30, 0), dpd)
    prior_dpd = np.where((dpd == 0) & (rng.random(n_loans) < 0.03), rng.integers(30, 60, n_loans), prior_dpd)

    g = current / (N_GRADES - 1)
    df["orig_grade"] = orig + 1
    df["prior_grade"] = prior + 1
    df["current_grade"] = current + 1
    df["dpd"] = dpd
    df["prior_dpd"] = prior_dpd
    df["watchlist"] = (rng.random(n_loans) < 0.01 + 0.10 * g ** 2).astype(int)
    df["forborne"] = (rng.random(n_loans) < 0.005 + 0.05 * g ** 2).astype(int)
    # Unlikely-to-pay defaults that are not (yet) 90 DPD
    df["default_flag"] = (rng.random(n_loans) < 0.002 + 0.02 * g ** 3).astype(int)
    return df


def load_portfolio(path: str) -> pd.DataFrame:
    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    return prepare_portfolio(df)


def prepare_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """Clean common file-format artefacts, then validate."""
    df = df.copy()
    # "None" is read as missing by pandas
    df["collateral_type"] = df["collateral_type"].fillna("None") if "collateral_type" in df else df.get("collateral_type")
    for col in ("watchlist", "forborne", "default_flag", "undrawn_limit", "collateral_value", "prior_dpd"):
        if col in df:
            df[col] = df[col].fillna(0)
    validate_portfolio(df)
    return df


def validate_portfolio(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Portfolio is missing required columns: {missing}")
    for col in ("orig_grade", "current_grade", "prior_grade"):
        if not df[col].between(1, N_GRADES).all():
            raise ValueError(f"{col} must be an integer between 1 and {N_GRADES}")
    from .config import COLLATERAL_PARAMS, SEGMENTS
    bad_seg = set(df["segment"]) - set(SEGMENTS)
    if bad_seg:
        raise ValueError(f"Unknown segment values: {bad_seg}; expected {SEGMENTS}")
    bad_coll = set(df["collateral_type"]) - set(COLLATERAL_PARAMS)
    if bad_coll:
        raise ValueError(f"Unknown collateral_type values: {bad_coll}; expected {list(COLLATERAL_PARAMS)}")
    bad_type = set(df["facility_type"]) - {"amortising", "bullet", "revolving"}
    if bad_type:
        raise ValueError(f"Unknown facility_type values: {bad_type}")
    if (df["drawn_balance"] < 0).any() or (df["undrawn_limit"] < 0).any():
        raise ValueError("Balances and limits must be non-negative")
