"""
Stage allocation and Significant Increase in Credit Risk (SICR).

Stage 3 (credit-impaired)  : DPD >= 90 (rebuttable presumption) OR default flag
Stage 2 (SICR)             : any of
    - DPD_30       : DPD >= 30 backstop
    - PD_RELATIVE  : lifetime PD now / lifetime PD expected at origination for the
                     same remaining window >= relative threshold AND absolute
                     increase >= absolute threshold (quantitative test)
    - DOWNGRADE    : rating downgrade >= N notches since origination
    - WATCHLIST    : on credit watchlist (qualitative)
    - FORBEARANCE  : forbearance / restructuring granted (qualitative)
Low credit risk exemption (IFRS 9 5.5.10): if enabled, loans whose 12m PD is
below the threshold skip the quantitative and downgrade tests. DPD and
qualitative backstops still apply.

Every loan carries the list of triggers that fired so the allocation is auditable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StagingPolicy


def assign_stages(
    df: pd.DataFrame,
    lifetime_pd_now: np.ndarray,
    pd_12m_now: np.ndarray,
    lifetime_pd_orig: np.ndarray,
    policy: StagingPolicy,
    grade_col: str = "current_grade",
    dpd_col: str = "dpd",
    use_flags: bool = True,
) -> pd.DataFrame:
    dpd = df[dpd_col].to_numpy()
    n = len(df)
    flags_default = df["default_flag"].to_numpy().astype(bool) if use_flags else np.zeros(n, bool)
    watch = df["watchlist"].to_numpy().astype(bool) if (use_flags and policy.use_watchlist) else np.zeros(n, bool)
    forb = df["forborne"].to_numpy().astype(bool) if (use_flags and policy.use_forbearance) else np.zeros(n, bool)

    s3_dpd = dpd >= policy.stage3_dpd
    s3 = s3_dpd | flags_default

    lcr = (pd_12m_now < policy.low_credit_risk_12m_pd) if policy.low_credit_risk_exemption else np.zeros(n, bool)
    ratio = lifetime_pd_now / np.maximum(lifetime_pd_orig, 1e-9)
    t_pd = (ratio >= policy.pd_relative_threshold) & \
           ((lifetime_pd_now - lifetime_pd_orig) >= policy.pd_absolute_threshold) & ~lcr
    notches = df[grade_col].to_numpy() - df["orig_grade"].to_numpy()
    t_notch = (notches >= policy.notch_downgrade_threshold) & ~lcr
    t_dpd30 = dpd >= policy.stage2_dpd

    s2 = ~s3 & (t_dpd30 | t_pd | t_notch | watch | forb)
    stage = np.where(s3, 3, np.where(s2, 2, 1))

    labels = []
    for i in range(n):
        if s3[i]:
            tr = [t for t, f in (("DPD_90", s3_dpd[i]), ("DEFAULT_FLAG", flags_default[i])) if f]
        elif s2[i]:
            tr = [t for t, f in (("DPD_30", t_dpd30[i]), ("PD_RELATIVE", t_pd[i]),
                                 ("DOWNGRADE", t_notch[i]), ("WATCHLIST", watch[i]),
                                 ("FORBEARANCE", forb[i])) if f]
        else:
            tr = ["LOW_CREDIT_RISK"] if lcr[i] else []
        labels.append(",".join(tr))

    return pd.DataFrame({
        "stage": stage,
        "sicr_triggers": labels,
        "pd_ratio_vs_origination": ratio,
        "notches_since_origination": notches,
        "low_credit_risk": lcr,
    }, index=df.index)
