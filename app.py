"""
IFRS 9 ECL Engine — interactive dashboard.

    streamlit run app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ecl_engine import DEFAULT_SCENARIOS, EngineConfig, StagingPolicy, generate_portfolio, reporting, run_ecl
from ecl_engine.config import GRADES, MACRO_VARIABLES, SEGMENTS
from ecl_engine.data import REQUIRED_COLUMNS, prepare_portfolio
from ecl_engine.macro import scenario_table
from ecl_engine.pd_model import pd_term_structure_table

st.set_page_config(page_title="IFRS 9 ECL Engine", page_icon="📉", layout="wide")

STAGE_COLORS = {1: "#2E7D5B", 2: "#D9A21B", 3: "#B23A3A"}
SCEN_COLORS = {"Upside": "#2E7D5B", "Base": "#3F6FB5", "Downside": "#B23A3A"}


def bn(x: float) -> str:
    return f"₦{x / 1e9:,.2f}bn"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Portfolio")
    source = st.radio("Data source", ["Synthetic loan book", "Upload loan tape"], label_visibility="collapsed")
    uploaded = None
    if source == "Synthetic loan book":
        n_loans = st.slider("Facilities", 1000, 20000, 5000, step=1000)
        seed = st.number_input("Random seed", value=42, step=1)
    else:
        uploaded = st.file_uploader("CSV or XLSX", type=["csv", "xlsx"])
        with st.expander("Required columns"):
            st.dataframe(pd.Series(REQUIRED_COLUMNS, name="description"), width="stretch")

    st.header("Scenario weights")
    w_up = st.slider("Upside", 0.0, 1.0, DEFAULT_SCENARIOS["Upside"]["weight"], 0.05)
    w_base = st.slider("Base", 0.0, 1.0, DEFAULT_SCENARIOS["Base"]["weight"], 0.05)
    w_down = st.slider("Downside", 0.0, 1.0, DEFAULT_SCENARIOS["Downside"]["weight"], 0.05)
    tot = w_up + w_base + w_down
    if tot == 0:
        st.error("At least one weight must be positive.")
        st.stop()
    st.caption(f"Normalised: {w_up/tot:.0%} / {w_base/tot:.0%} / {w_down/tot:.0%}")

    st.header("SICR policy")
    rel = st.slider("Relative lifetime PD increase (×)", 1.5, 4.0, 2.5, 0.25)
    abs_pp = st.slider("Absolute lifetime PD increase (pp)", 0.0, 5.0, 1.0, 0.25)
    notch = st.slider("Downgrade backstop (notches)", 1, 5, 3)
    dpd2 = st.select_slider("Stage 2 DPD backstop", [30, 45, 60], 30)
    dpd3 = st.select_slider("Stage 3 DPD backstop", [90, 120, 180], 90)
    use_watch = st.checkbox("Watchlist ⇒ Stage 2", True)
    use_forb = st.checkbox("Forbearance ⇒ Stage 2", True)
    lcr = st.checkbox("Low credit risk exemption", True)
    lcr_pd = st.slider("LCR 12m PD threshold (%)", 0.1, 1.0, 0.3, 0.05, disabled=not lcr)


@st.cache_data(show_spinner=False)
def get_portfolio(n: int, s: int) -> pd.DataFrame:
    return generate_portfolio(n, s)


@st.cache_data(show_spinner="Running ECL engine…")
def compute(df: pd.DataFrame, weights: tuple, policy: dict):
    cfg = EngineConfig()
    for name, w in zip(["Upside", "Base", "Downside"], weights):
        cfg.scenarios[name]["weight"] = w
    cfg.staging = StagingPolicy(**policy)
    return run_ecl(df, cfg)


if uploaded is not None:
    raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
    try:
        portfolio = prepare_portfolio(raw)
    except (ValueError, KeyError) as e:
        st.error(str(e))
        st.stop()
elif source == "Upload loan tape":
    st.info("Upload a loan tape in the sidebar, or switch to the synthetic loan book.")
    st.stop()
else:
    portfolio = get_portfolio(n_loans, int(seed))

policy = dict(pd_relative_threshold=rel, pd_absolute_threshold=abs_pp / 100, notch_downgrade_threshold=notch,
              stage2_dpd=dpd2, stage3_dpd=dpd3, use_watchlist=use_watch, use_forbearance=use_forb,
              low_credit_risk_exemption=lcr, low_credit_risk_12m_pd=lcr_pd / 100)
res = compute(portfolio, (w_up, w_base, w_down), policy)
L = res.loans
h = reporting.headline(res)

# ------------------------------------------------------------------ header
st.title("IFRS 9 Expected Credit Loss Engine")
st.caption("Staging & SICR · lifetime PD term structures · collateral LGD · EAD profiles · "
           "probability-weighted macro scenarios. Synthetic data, illustrative calibration.")

c = st.columns(5)
c[0].metric("Total ECL", bn(h["ecl"]))
c[1].metric("EAD", bn(h["ead"]))
c[2].metric("Coverage", f"{h['coverage_%']:.2f}%")
c[3].metric("Stage 2 share (EAD)", f"{h['stage2_share_%']:.1f}%")
c[4].metric("Stage 3 share (EAD)", f"{h['stage3_share_%']:.1f}%")

tabs = st.tabs(["Overview", "Staging & SICR", "PD term structure", "LGD & EAD", "Macro scenarios",
                "Loan drill-down", "Methodology"])

# ------------------------------------------------------------------ overview
with tabs[0]:
    ss = reporting.stage_summary(res, by="segment")
    left, right = st.columns(2)
    fig = px.bar(ss, x="segment", y="ecl", color=ss["stage"].astype(str), barmode="stack",
                 color_discrete_map={str(k): v for k, v in STAGE_COLORS.items()},
                 labels={"ecl": "ECL (₦)", "color": "Stage"}, title="ECL by segment and stage")
    left.plotly_chart(fig, width="stretch")
    fig = px.bar(ss, x="segment", y="coverage_%", color=ss["stage"].astype(str), barmode="group",
                 color_discrete_map={str(k): v for k, v in STAGE_COLORS.items()},
                 labels={"coverage_%": "Coverage %", "color": "Stage"}, title="Coverage ratio (ECL / EAD)")
    right.plotly_chart(fig, width="stretch")

    st.subheader("Disclosure table — by stage")
    t = reporting.stage_summary(res)
    st.dataframe(t.style.format({"gross_carrying_amount": "{:,.0f}", "ead": "{:,.0f}", "ecl": "{:,.0f}",
                                 "coverage_%": "{:.2f}", "share_of_ead_%": "{:.1f}"}),
                 width="stretch", hide_index=True)
    st.subheader("By product")
    p = reporting.stage_summary(res, by="product")
    st.dataframe(p.style.format({"gross_carrying_amount": "{:,.0f}", "ead": "{:,.0f}", "ecl": "{:,.0f}",
                                 "coverage_%": "{:.2f}", "share_of_ead_%": "{:.1f}"}),
                 width="stretch", hide_index=True)

# ------------------------------------------------------------------ staging
with tabs[1]:
    left, right = st.columns(2)
    mm = reporting.migration_matrix(res)
    fig = px.imshow(mm / 1e9, text_auto=".1f", color_continuous_scale="Blues", aspect="auto",
                    title="Stage migration, prior → current (EAD ₦bn)")
    left.plotly_chart(fig, width="stretch")
    ts = reporting.trigger_summary(res)
    fig = px.bar(ts, x="loans", y="trigger", color=ts["stage"].astype(str), orientation="h",
                 color_discrete_map={str(k): v for k, v in STAGE_COLORS.items()},
                 title="Triggers fired (a loan can hit several)", labels={"color": "Stage"})
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    right.plotly_chart(fig, width="stretch")
    st.dataframe(ts, width="stretch", hide_index=True)

    st.subheader("Quantitative SICR test")
    perf = L[L.stage < 3].copy()
    perf["Stage"] = perf["stage"].astype(str)
    fig = px.scatter(perf.sample(min(3000, len(perf)), random_state=1),
                     x="pd_lifetime_at_origination", y="pd_lifetime", color="Stage", log_x=True, log_y=True,
                     color_discrete_map={str(k): v for k, v in STAGE_COLORS.items()}, opacity=0.6,
                     hover_data=["loan_id", "product", "orig_grade", "current_grade", "sicr_triggers"],
                     labels={"pd_lifetime_at_origination": "Lifetime PD expected at origination",
                             "pd_lifetime": "Lifetime PD now (scenario-weighted)"})
    xs = np.logspace(np.log10(max(perf["pd_lifetime_at_origination"].min(), 1e-5)), 0, 50)
    fig.add_trace(go.Scatter(x=xs, y=np.minimum(xs * rel, 1), mode="lines", name=f"{rel}× threshold",
                             line=dict(dash="dash", color="black")))
    st.plotly_chart(fig, width="stretch")

    cl = reporting.cliff_effect(res).iloc[0]
    st.info(f"**Stage 2 cliff effect:** measuring Stage 2 on a lifetime basis adds {bn(cl['lifetime_uplift'])} "
            f"over a 12-month view ({cl['multiple']:.2f}× the 12-month ECL).")

# ------------------------------------------------------------------ PD
with tabs[2]:
    seg = st.selectbox("Segment", SEGMENTS, index=1)
    scen = st.radio("Conditioning", ["TTC", "Upside", "Base", "Downside"], horizontal=True, index=2)
    z = None if scen == "TTC" else res.z_paths[scen]
    pts = pd_term_structure_table(seg, 10, z)
    long = pts.reset_index().melt(id_vars="grade", var_name="year", value_name="cumulative_pd")
    fig = px.line(long, x="year", y="cumulative_pd", color="grade", markers=True,
                  title=f"Cumulative lifetime PD by grade — {seg}, {scen}",
                  labels={"cumulative_pd": "Cumulative PD", "year": "Horizon (years)"})
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    st.dataframe((pts * 100).round(2).add_suffix("y"), width="stretch")
    st.caption("Built from Markov powers of the annual migration matrix; macro conditioning applied to "
               "annual hazards through the Z-shift, reverting to TTC after year 3 + reversion period.")

# ------------------------------------------------------------------ LGD / EAD
with tabs[3]:
    left, right = st.columns(2)
    lg = L[L.stage < 3].groupby(["product", "collateral_type"]).apply(
        lambda g: pd.Series({"lgd": np.average(g["lgd_today"], weights=g["ead_today"]), "ead": g["ead_today"].sum()}),
        include_groups=False).reset_index()
    fig = px.bar(lg, x="product", y="lgd", color="collateral_type", barmode="group",
                 title="EAD-weighted LGD by product and collateral", labels={"lgd": "LGD"})
    fig.update_yaxes(tickformat=".0%")
    left.plotly_chart(fig, width="stretch")
    fig = px.histogram(L, x="lgd_today", color="segment", nbins=40, title="LGD distribution", barmode="overlay")
    right.plotly_chart(fig, width="stretch")

    st.subheader("Exposure profile of a single facility")
    lid = st.selectbox("Facility", L.sort_values("ead_today", ascending=False)["loan_id"].head(300))
    from ecl_engine.ead import ead_schedule
    row = L[L.loan_id == lid]
    e = ead_schedule(row, res.config.max_years * res.config.steps_per_year)[0]
    nz = max(int((e > 0).sum()), 1)
    fig = px.area(x=np.arange(nz) / 4, y=e[:nz], labels={"x": "Years from reporting date", "y": "EAD (₦)"},
                  title=f"{lid} — {row['product'].iloc[0]} ({row['facility_type'].iloc[0]})")
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ macro
with tabs[4]:
    sc = reporting.scenario_summary(res)
    left, right = st.columns([1, 1])
    fig = px.bar(sc.reset_index(), x="index", y="ecl", color="index",
                 color_discrete_map={**SCEN_COLORS, "Probability-weighted": "#555"},
                 title="ECL under each scenario vs probability-weighted", labels={"index": "", "ecl": "ECL (₦)"})
    fig.update_layout(showlegend=False)
    left.plotly_chart(fig, width="stretch")
    zdf = pd.DataFrame({k: v[:8] for k, v in res.z_paths.items()}, index=range(1, 9))
    fig = px.line(zdf, markers=True, color_discrete_map=SCEN_COLORS,
                  title="Credit-cycle index Z (reverts to 0 = TTC)", labels={"index": "Year", "value": "Z"})
    right.plotly_chart(fig, width="stretch")
    nl = sc.loc["Probability-weighted", "ecl"] - sc.loc["Base", "ecl"]
    st.info(f"**Non-linearity adjustment:** the probability-weighted ECL is {bn(nl)} "
            f"({sc.loc['Probability-weighted', 'vs_base_%']:.1f}%) above the Base-only ECL, because losses rise "
            "faster in the downside than they fall in the upside (IFRS 9 5.5.17(a)).")
    st.subheader("Scenario macro paths")
    st.dataframe(scenario_table(res.config.scenarios, res.weights), width="stretch", hide_index=True)
    st.caption("Satellite model: Z = Σ βₖ·(xₖ − μₖ)/σₖ. " + " · ".join(
        f"{k}: μ={m}, σ={s}, β={b:+}" for k, (m, s, b) in MACRO_VARIABLES.items()))

# ------------------------------------------------------------------ drill-down
with tabs[5]:
    f1, f2, f3 = st.columns(3)
    seg_f = f1.multiselect("Segment", SEGMENTS, SEGMENTS)
    stg_f = f2.multiselect("Stage", [1, 2, 3], [2, 3])
    q = f3.text_input("Loan ID contains")
    view = L[L.segment.isin(seg_f) & L.stage.isin(stg_f)]
    if q:
        view = view[view.loan_id.str.contains(q, case=False)]
    cols = ["loan_id", "segment", "product", "orig_grade", "current_grade", "dpd", "stage", "prior_stage",
            "sicr_triggers", "pd_12m", "pd_lifetime", "pd_lifetime_at_origination", "lgd_today",
            "ead_today", "ecl", "coverage"]
    st.dataframe(view[cols].sort_values("ecl", ascending=False).style.format(
        {"pd_12m": "{:.2%}", "pd_lifetime": "{:.2%}", "pd_lifetime_at_origination": "{:.2%}",
         "lgd_today": "{:.1%}", "ead_today": "{:,.0f}", "ecl": "{:,.0f}", "coverage": "{:.2%}"}),
        width="stretch", hide_index=True, height=520)

    with tempfile.NamedTemporaryFile(suffix=".xlsx") as tmp:
        reporting.export_excel(res, tmp.name)
        data = open(tmp.name, "rb").read()
    st.download_button("⬇ Download full results (Excel)", data, "ifrs9_ecl_results.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ------------------------------------------------------------------ methodology
with tabs[6]:
    st.markdown((Path(__file__).parent / "docs" / "METHODOLOGY.md").read_text(encoding="utf-8"))
