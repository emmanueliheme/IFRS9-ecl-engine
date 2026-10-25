"""Disclosure-style summaries and Excel export."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ecl import ECLResult
from .macro import scenario_table


def stage_summary(res: ECLResult, by: str | None = None) -> pd.DataFrame:
    keys = ([by] if by else []) + ["stage"]
    g = res.loans.groupby(keys).agg(
        loans=("loan_id", "count"),
        gross_carrying_amount=("drawn_balance", "sum"),
        ead=("ead_today", "sum"),
        ecl=("ecl", "sum"),
    )
    g["coverage_%"] = 100 * g["ecl"] / g["ead"]
    g["share_of_ead_%"] = 100 * g["ead"] / g.groupby(level=0)["ead"].transform("sum") if by else \
        100 * g["ead"] / g["ead"].sum()
    return g.reset_index()


def migration_matrix(res: ECLResult, weight: str = "ead_today") -> pd.DataFrame:
    m = res.loans.pivot_table(index="prior_stage", columns="stage", values=weight,
                              aggfunc="sum", fill_value=0.0)
    m = m.reindex(index=[1, 2, 3], columns=[1, 2, 3], fill_value=0.0)
    m.index = [f"From Stage {i}" for i in m.index]
    m.columns = [f"To Stage {c}" for c in m.columns]
    return m


def trigger_summary(res: ECLResult) -> pd.DataFrame:
    s = res.loans.loc[res.loans["stage"] >= 2, ["stage", "sicr_triggers", "ead_today"]].copy()
    s["trigger"] = s["sicr_triggers"].str.split(",")
    s = s.explode("trigger")
    out = s.groupby(["stage", "trigger"]).agg(loans=("trigger", "size"), ead=("ead_today", "sum")).reset_index()
    sole = res.loans.loc[res.loans["stage"] >= 2, "sicr_triggers"]
    sole = sole[~sole.str.contains(",")].value_counts().rename("sole_trigger_loans")
    return out.merge(sole, left_on="trigger", right_index=True, how="left").fillna({"sole_trigger_loans": 0})


def scenario_summary(res: ECLResult) -> pd.DataFrame:
    tot = res.scenario_ecl.sum()
    out = pd.DataFrame({"weight": pd.Series(res.weights), "ecl": tot})
    out.loc["Probability-weighted"] = [1.0, res.loans["ecl"].sum()]
    base = out.loc["Base", "ecl"] if "Base" in out.index else np.nan
    out["vs_base_%"] = 100 * (out["ecl"] / base - 1)
    return out


def cliff_effect(res: ECLResult) -> pd.DataFrame:
    """How much ECL is due to Stage 2 lifetime measurement vs a 12-month view."""
    s2 = res.loans[res.loans["stage"] == 2]
    return pd.DataFrame({
        "stage2_ead": [s2["ead_today"].sum()],
        "ecl_if_12m": [s2["ecl_12m"].sum()],
        "ecl_lifetime": [s2["ecl"].sum()],
        "lifetime_uplift": [s2["ecl"].sum() - s2["ecl_12m"].sum()],
        "multiple": [s2["ecl"].sum() / max(s2["ecl_12m"].sum(), 1e-9)],
    })


def headline(res: ECLResult) -> dict:
    l = res.loans
    return {
        "loans": len(l),
        "gross_carrying_amount": l["drawn_balance"].sum(),
        "ead": l["ead_today"].sum(),
        "ecl": l["ecl"].sum(),
        "coverage_%": 100 * l["ecl"].sum() / l["ead_today"].sum(),
        "stage2_share_%": 100 * l.loc[l.stage == 2, "ead_today"].sum() / l["ead_today"].sum(),
        "stage3_share_%": 100 * l.loc[l.stage == 3, "ead_today"].sum() / l["ead_today"].sum(),
    }


def export_excel(res: ECLResult, path: str) -> str:
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame([headline(res)]).T.rename(columns={0: "value"}).to_excel(xw, sheet_name="Headline")
        stage_summary(res).to_excel(xw, sheet_name="By Stage", index=False)
        stage_summary(res, by="segment").to_excel(xw, sheet_name="By Segment", index=False)
        stage_summary(res, by="product").to_excel(xw, sheet_name="By Product", index=False)
        migration_matrix(res).to_excel(xw, sheet_name="Stage Migration")
        trigger_summary(res).to_excel(xw, sheet_name="SICR Triggers", index=False)
        scenario_summary(res).to_excel(xw, sheet_name="Scenarios")
        scenario_table(res.config.scenarios, res.weights).to_excel(xw, sheet_name="Macro Paths", index=False)
        cliff_effect(res).to_excel(xw, sheet_name="Stage 2 Cliff", index=False)
        res.loans.to_excel(xw, sheet_name="Loan Level", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 40)
            ws.freeze_panes = "A2"
    return path
