"""
Command-line runner.

    python run_ecl.py                         # synthetic 5,000-loan book
    python run_ecl.py --portfolio my.csv      # your own data (see REQUIRED_COLUMNS)
    python run_ecl.py --weights 0.25 0.45 0.30 --out outputs/ecl.xlsx
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ecl_engine import EngineConfig, generate_portfolio, load_portfolio, reporting, run_ecl


def main() -> None:
    ap = argparse.ArgumentParser(description="IFRS 9 ECL engine")
    ap.add_argument("--portfolio", help="CSV/XLSX loan tape; omit to use synthetic data")
    ap.add_argument("--n-loans", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weights", type=float, nargs=3, metavar=("UPSIDE", "BASE", "DOWNSIDE"))
    ap.add_argument("--out", default="outputs/ecl_results.xlsx")
    args = ap.parse_args()

    df = load_portfolio(args.portfolio) if args.portfolio else generate_portfolio(args.n_loans, args.seed)
    cfg = EngineConfig()
    if args.weights:
        for name, w in zip(["Upside", "Base", "Downside"], args.weights):
            cfg.scenarios[name]["weight"] = w
    res = run_ecl(df, cfg)

    h = reporting.headline(res)
    print("\nIFRS 9 ECL RUN")
    print("=" * 52)
    print(f"Facilities            : {h['loans']:,}")
    print(f"EAD (NGN bn)          : {h['ead'] / 1e9:,.2f}")
    print(f"ECL (NGN bn)          : {h['ecl'] / 1e9:,.2f}")
    print(f"Coverage              : {h['coverage_%']:.2f}%")
    print(f"Stage 2 / 3 share EAD : {h['stage2_share_%']:.1f}% / {h['stage3_share_%']:.1f}%")
    print("\nBy stage")
    print(reporting.stage_summary(res).round(2).to_string(index=False))
    print("\nScenarios")
    print(reporting.scenario_summary(res).round(2).to_string())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    reporting.export_excel(res, args.out)
    res.loans.to_csv(Path(args.out).with_suffix(".csv"), index=False)
    print(f"\nSaved {args.out} and {Path(args.out).with_suffix('.csv')}")


if __name__ == "__main__":
    main()
