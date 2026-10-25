# Methodology

## 1. Measurement basis

| Stage | Definition | Measurement |
|---|---|---|
| 1 | Performing, no SICR since initial recognition | 12-month ECL |
| 2 | Significant increase in credit risk (SICR) | Lifetime ECL |
| 3 | Credit-impaired | Lifetime ECL, PD = 100% |

For each scenario *s*, on a quarterly grid *q* over the exposure life:

```
ECL_s = Σ_q  MarginalPD_s(q) × LGD_s(q) × EAD(q) × (1 + EIR)^-(t_q + ¼)
MarginalPD_s(q) = Survival_s(t_q) × Hazard_s(q)
ECL = Σ_s  w_s × ECL_s
```

Stage 1 sums the first four quarters only; Stage 2 sums the full exposure life; Stage 3 is `EAD_today × LGD_s(0)`. Cash shortfalls are discounted at the original effective interest rate (IFRS 9 B5.5.44).

## 2. Staging and SICR

Staging is done once, on scenario-weighted PDs, so a facility has one stage regardless of scenario.

**Stage 3** — `DPD ≥ 90` (rebuttable presumption, B5.5.37) **or** a qualitative unlikely-to-pay default flag.

**Stage 2** — any trigger below fires (and Stage 3 does not):

| Trigger | Rule | Type |
|---|---|---|
| `PD_RELATIVE` | Lifetime PD now ÷ lifetime PD expected at origination for the same remaining window ≥ 2.5× **and** absolute increase ≥ 1pp | Quantitative |
| `DOWNGRADE` | ≥ 3 notches below origination grade | Quantitative backstop |
| `DPD_30` | DPD ≥ 30 (rebuttable presumption, 5.5.11) | Backstop |
| `WATCHLIST` | On credit watchlist | Qualitative |
| `FORBEARANCE` | Forbearance / restructuring granted | Qualitative |

**Low credit risk exemption** (5.5.10) — when enabled, facilities with 12-month PD below 0.3% skip the PD and downgrade tests; DPD and qualitative backstops still apply.

The origination benchmark is a *forward* PD: the TTC cumulative PD of the origination grade between the facility's current age and its maturity, so that like is compared with like over the same remaining life (B5.5.11).

Every facility stores the list of triggers that fired, and a prior-period stage is reconstructed from prior ratings and DPD to produce the stage migration matrix.

## 3. Lifetime PD

1. **Migration matrix.** An annual 9×9 through-the-cycle matrix (8 grades + absorbing default) per segment. The engine ships assumption-based matrices and a cohort estimator (`estimate_transition_matrix`) to replace them with observed migrations.
2. **TTC term structure.** Cumulative PD to horizon *t* is the default column of *M^t*, so rating migration over the life of the loan is captured.
3. **Conditional hazards.** `h_t = (CPD_t − CPD_{t−1}) / (1 − CPD_{t−1})`.
4. **Point-in-time conditioning.** Each annual hazard is shifted by the scenario credit index Z on the probit scale:

   `h_PIT = Φ( Φ⁻¹(h_TTC) − Z · √(ρ / (1 − ρ)) )`

   ρ is the Vasicek asset correlation (Retail 0.08, SME 0.12, Corporate 0.18). The form returns the TTC hazard exactly when Z = 0.
5. **Quarterly split.** `h_q = 1 − (1 − h_annual)^¼`.

## 4. Forward-looking macro scenarios

A satellite model maps standardised macro variables to Z:

`Z_t = Σ_k β_k · (x_{k,t} − μ_k) / σ_k`

| Variable | μ | σ | β |
|---|---|---|---|
| Real GDP growth (%) | 3.0 | 1.5 | +0.30 |
| Inflation (%) | 18.0 | 6.0 | −0.15 |
| Brent oil (USD/bbl) | 75 | 15 | +0.20 |
| Monetary policy rate (%) | 20.0 | 5.0 | −0.12 |

Three scenarios (Upside 20%, Base 50%, Downside 30%) carry explicit paths for years 1–3. Z then reverts linearly to 0 over two years, reflecting that forecasts beyond a reasonable and supportable horizon revert to long-run information (B5.5.54). Weights are editable in the dashboard and normalised to 100%.

Because losses are convex in Z, the probability-weighted ECL exceeds the Base-only ECL. The dashboard reports this **non-linearity adjustment** explicitly.

## 5. LGD

```
RecoverablePV_t = CollateralValue × (1 + γ_c · Z_t) × (1 − haircut_c) / (1 + EIR)^RecoveryYears_c
Shortfall_t     = max(0, 1 − RecoverablePV_t / EAD_t)
LGD_t           = max(floor, (1 − CureRate) × Shortfall_t × UnsecuredLGD × (1 − δ · Z_t))
```

| Collateral | Haircut | Time to recovery | γ (Z sensitivity) |
|---|---|---|---|
| Property | 30% | 3.0y | 0.10 |
| Equipment | 45% | 1.5y | 0.08 |
| Vehicle | 40% | 1.0y | 0.08 |
| Cash | 5% | 0.25y | 0.00 |

| Segment | Unsecured LGD | Cure rate | Floor | δ |
|---|---|---|---|---|
| Retail | 75% | 20% | 5% | 0.08 |
| SME | 65% | 15% | 5% | 0.10 |
| Corporate | 55% | 10% | 5% | 0.10 |

LGD is time-varying: as an amortising loan runs down, the same collateral covers a larger share of EAD.

## 6. EAD

| Facility | Profile |
|---|---|
| Amortising | Annuity schedule at the EIR over the remaining term, with a constant prepayment rate (personal loans 8%, mortgages 4% p.a.) |
| Bullet | Drawn balance until maturity |
| Revolving | Drawn + CCF × undrawn (cards 65%, SME overdrafts 50%), flat over a behavioural life of 3y (cards) / 2y (overdrafts) rather than the contractual notice period (5.5.20) |

## 7. Limitations and next steps

- Calibrations are illustrative. A production model needs migration matrices, cure rates, haircuts, CCFs and the macro satellite model fitted to the bank's own data and validated.
- The origination benchmark uses TTC rather than the PIT lifetime PD actually recorded at origination; a production system should store the origination PD term structure per facility.
- No probation/cure period for exits from Stage 3 or Stage 2 is applied; this would require facility-level stage history.
- Single systematic factor; segment-specific satellite models would improve sensitivity.
- Post-model adjustments (management overlays) are out of scope but would sit on top of the modelled ECL with their own governance.
