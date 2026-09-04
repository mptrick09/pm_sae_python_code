"""
Step 4: Run the full analysis end to end.

Usage:
    python -m src.step4_run_analysis --data data/analysis_dataset.pkl --out results/

Requires a dataset built by step1_build_dataset.py (real data, not included
in this repo -- see README) OR the synthetic dataset from
tests/generate_synthetic_data.py for a dry run without real data.
"""
import argparse
import numpy as np
import pandas as pd

from src.step2_models import fit_bhf, fit_pm, lambda_test
from src.step3_poverty_indicators import district_level_indicators

COVARIATES = ["hhsize", "head_age", "head_female", "electricity", "log_rooms", "log_livestock"]


def prepare_covariates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["head_female"] = (df["head_sex"] == 2).astype(int)
    df["electricity"] = (df["electricity_grid"] == 1).astype(int)
    df["log_rooms"] = np.log(df["n_rooms"].clip(lower=1))
    df["log_livestock"] = np.log1p(df["livestock_index"])
    df["district"] = df["district"].astype(int).astype(str)
    return df.dropna(subset=COVARIATES + ["welfare", "district", "pop_wt"])


def run(data_path: str, welfare_col: str, poverty_line: float, out_dir: str):
    df = pd.read_pickle(data_path)
    if welfare_col != "welfare":
        df = df.rename(columns={welfare_col: "welfare"})
    df = prepare_covariates(df)
    print(f"Analysis sample: {len(df)} households, {df['district'].nunique()} districts")

    print("\n=== Fitting BHF ===")
    bhf_fit = fit_bhf(df, "welfare", COVARIATES, "district")
    print(bhf_fit.summary())

    print("\n=== Fitting PM ===")
    pm_fit, knots, spline_cols = fit_pm(df, "welfare", COVARIATES, "district", "hhsize")
    print(pm_fit.summary())

    print("\n=== Lambda test (H0: lambda = 0, i.e. BHF is adequate) ===")
    sigma_g2 = pm_fit.vcomp[0] if len(pm_fit.vcomp) > 0 else 0.0
    sigma_e2 = pm_fit.scale
    lam = sigma_g2 / (sigma_g2 + sigma_e2)
    lr_stat, p_val = lambda_test(bhf_fit, pm_fit)
    print(f"lambda = {lam:.5f}, LR = {lr_stat:.3f}, boundary-adjusted p = {p_val:.4f}")

    print("\n=== District-level poverty indicators ===")
    bhf_dist = district_level_indicators(df, bhf_fit.fittedvalues.values, np.sqrt(bhf_fit.scale),
                                          "pop_wt", "district", poverty_line)
    pm_dist = district_level_indicators(df, pm_fit.fittedvalues.values, np.sqrt(pm_fit.scale),
                                         "pop_wt", "district", poverty_line)
    comp = bhf_dist[["HCR", "PG"]].join(pm_dist[["HCR", "PG"]], lsuffix="_BHF", rsuffix="_PM")
    comp["HCR_diff"] = comp["HCR_PM"] - comp["HCR_BHF"]

    import os
    os.makedirs(out_dir, exist_ok=True)
    comp.to_csv(f"{out_dir}/district_comparison.csv")
    with open(f"{out_dir}/summary.txt", "w") as f:
        f.write(f"lambda={lam:.5f}\nLR={lr_stat:.3f}\np_value={p_val:.5f}\n")
        f.write(f"National HCR BHF={np.average(bhf_dist['HCR'], weights=bhf_dist['n_households']):.3f}\n")
        f.write(f"National HCR PM={np.average(pm_dist['HCR'], weights=pm_dist['n_households']):.3f}\n")
    print(f"\nResults written to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--welfare-col", default="sol_jan",
                         help="Deflated welfare variable (NOT cons1ae -- see README)")
    parser.add_argument("--poverty-line", type=float, default=560127)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    run(args.data, args.welfare_col, args.poverty_line, args.out)
