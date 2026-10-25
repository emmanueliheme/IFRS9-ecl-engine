"""
Central configuration for the IFRS 9 ECL engine.

Every number here is an ILLUSTRATIVE assumption chosen to make the engine
behave realistically on synthetic data. In production these would be
replaced by values calibrated on the bank's own history and approved
by the model governance / credit risk committee.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

# --------------------------------------------------------------------------
# Rating scale (8 performing grades + absorbing default state)
# --------------------------------------------------------------------------
GRADES: List[str] = ["1 (AAA/AA)", "2 (A)", "3 (BBB)", "4 (BB)", "5 (B+)", "6 (B)", "7 (B-)", "8 (CCC)"]
N_GRADES = len(GRADES)
DEFAULT_STATE = N_GRADES  # index of the absorbing "D" state

# Through-the-cycle 1-year default rates per grade (before segment scaling)
TTC_ANNUAL_PD = [0.0005, 0.0012, 0.0030, 0.0075, 0.0160, 0.0350, 0.0800, 0.2000]
# Probability of staying in the same grade over one year
STAY_PROB = [0.90, 0.88, 0.85, 0.80, 0.76, 0.72, 0.66, 0.60]

SEGMENTS = ["Retail", "SME", "Corporate"]


@dataclass
class SegmentParams:
    pd_multiplier: float      # scales the TTC default column of the migration matrix
    asset_correlation: float  # Vasicek rho used to condition PDs on the macro factor Z
    unsecured_lgd: float      # loss on an unsecured claim, before cure
    cure_rate: float          # share of defaults that return to performing with ~no loss
    lgd_floor: float          # regulatory/prudential LGD floor
    lgd_z_sensitivity: float  # unsecured LGD multiplier = 1 - sensitivity * Z


SEGMENT_PARAMS: Dict[str, SegmentParams] = {
    "Retail":    SegmentParams(1.30, 0.08, 0.75, 0.20, 0.05, 0.08),
    "SME":       SegmentParams(1.60, 0.12, 0.65, 0.15, 0.05, 0.10),
    "Corporate": SegmentParams(1.00, 0.18, 0.55, 0.10, 0.05, 0.10),
}

# --------------------------------------------------------------------------
# Collateral
# --------------------------------------------------------------------------
@dataclass
class CollateralParams:
    haircut: float          # forced-sale discount on market value
    recovery_years: float   # time from default to cash realisation
    z_sensitivity: float    # collateral value index = 1 + sensitivity * Z


COLLATERAL_PARAMS: Dict[str, CollateralParams] = {
    "Property":  CollateralParams(0.30, 3.0, 0.10),
    "Equipment": CollateralParams(0.45, 1.5, 0.08),
    "Vehicle":   CollateralParams(0.40, 1.0, 0.08),
    "Cash":      CollateralParams(0.05, 0.25, 0.00),
    "None":      CollateralParams(1.00, 0.0, 0.00),
}

# --------------------------------------------------------------------------
# Products
# --------------------------------------------------------------------------
@dataclass
class ProductParams:
    segment: str
    facility_type: str                 # amortising | bullet | revolving
    term_years: tuple                  # (min, max) original contractual term
    amount_median_ngn: float           # median drawn/limit amount
    collateral_mix: Dict[str, float]   # collateral type -> probability
    ltv_range: tuple = (0.4, 0.9)      # loan-to-value at origination for secured loans
    ccf: float = 0.0                   # credit conversion factor on undrawn (revolving)
    behavioural_life_years: float = 0.0  # used instead of contractual term for revolvers
    annual_prepayment: float = 0.0     # constant prepayment rate (amortising only)
    eir_range: tuple = (0.18, 0.30)
    mix: float = 0.0                   # share of portfolio count


PRODUCTS: Dict[str, ProductParams] = {
    "Personal Loan": ProductParams("Retail", "amortising", (1, 5), 2.5e6, {"None": 1.0},
                                   annual_prepayment=0.08, eir_range=(0.24, 0.34), mix=0.30),
    "Mortgage": ProductParams("Retail", "amortising", (10, 20), 35e6, {"Property": 1.0},
                              ltv_range=(0.5, 0.85), annual_prepayment=0.04, eir_range=(0.16, 0.24), mix=0.08),
    "Credit Card": ProductParams("Retail", "revolving", (1, 1), 1.5e6, {"None": 1.0},
                                 ccf=0.65, behavioural_life_years=3.0, eir_range=(0.30, 0.42), mix=0.22),
    "SME Term Loan": ProductParams("SME", "amortising", (2, 6), 40e6,
                                   {"Property": 0.45, "Equipment": 0.30, "None": 0.25},
                                   eir_range=(0.22, 0.32), mix=0.15),
    "SME Overdraft": ProductParams("SME", "revolving", (1, 1), 15e6,
                                   {"Property": 0.30, "Cash": 0.10, "None": 0.60},
                                   ccf=0.50, behavioural_life_years=2.0, eir_range=(0.24, 0.34), mix=0.12),
    "Corporate Term Loan": ProductParams("Corporate", "amortising", (3, 10), 1.5e9,
                                         {"Property": 0.50, "Equipment": 0.30, "None": 0.20},
                                         eir_range=(0.18, 0.27), mix=0.09),
    "Corporate Bullet": ProductParams("Corporate", "bullet", (1, 5), 2.5e9,
                                      {"Cash": 0.15, "Property": 0.35, "None": 0.50},
                                      eir_range=(0.17, 0.25), mix=0.04),
}

# --------------------------------------------------------------------------
# Forward-looking macroeconomic scenarios
# (illustrative paths for years 1-3, not forecasts)
# --------------------------------------------------------------------------
MACRO_VARIABLES = {
    # name: (long-run mean, std dev, coefficient in the Z satellite model)
    "gdp_growth":   (3.0, 1.5, +0.30),   # real GDP growth %, higher = better
    "inflation":    (18.0, 6.0, -0.15),  # headline CPI %, higher = worse
    "oil_price":    (75.0, 15.0, +0.20), # Brent USD/bbl, higher = better for an oil exporter
    "policy_rate":  (20.0, 5.0, -0.12),  # monetary policy rate %, higher = tighter
}

DEFAULT_SCENARIOS = {
    "Upside": {
        "weight": 0.20,
        "gdp_growth": [4.5, 5.0, 5.0], "inflation": [18.0, 14.0, 12.0],
        "oil_price": [90.0, 88.0, 85.0], "policy_rate": [22.0, 19.0, 17.0],
    },
    "Base": {
        "weight": 0.50,
        "gdp_growth": [3.4, 3.6, 3.8], "inflation": [22.0, 18.0, 15.0],
        "oil_price": [75.0, 75.0, 75.0], "policy_rate": [26.0, 24.0, 22.0],
    },
    "Downside": {
        "weight": 0.30,
        "gdp_growth": [1.0, 1.8, 2.5], "inflation": [30.0, 26.0, 22.0],
        "oil_price": [55.0, 60.0, 65.0], "policy_rate": [29.0, 28.0, 26.0],
    },
}

# Years over which Z reverts linearly from the last explicit forecast to 0 (TTC)
MEAN_REVERSION_YEARS = 2

# --------------------------------------------------------------------------
# Staging / SICR policy
# --------------------------------------------------------------------------
@dataclass
class StagingPolicy:
    stage3_dpd: int = 90                 # rebuttable presumption, IFRS 9 B5.5.37
    stage2_dpd: int = 30                 # rebuttable presumption, IFRS 9 5.5.11
    pd_relative_threshold: float = 2.5   # current lifetime PD / origination lifetime PD
    pd_absolute_threshold: float = 0.01  # and an absolute increase of at least 1pp
    notch_downgrade_threshold: int = 3   # qualitative backstop on rating migration
    use_watchlist: bool = True
    use_forbearance: bool = True
    low_credit_risk_exemption: bool = True
    low_credit_risk_12m_pd: float = 0.003  # roughly investment grade


@dataclass
class EngineConfig:
    scenarios: dict = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_SCENARIOS.items()})
    staging: StagingPolicy = field(default_factory=StagingPolicy)
    max_years: int = 25          # horizon of the PD term structure grid
    steps_per_year: int = 4      # quarterly time grid

    def normalised_weights(self) -> Dict[str, float]:
        total = sum(s["weight"] for s in self.scenarios.values())
        if total <= 0:
            raise ValueError("Scenario weights must sum to a positive number")
        return {k: s["weight"] / total for k, s in self.scenarios.items()}
