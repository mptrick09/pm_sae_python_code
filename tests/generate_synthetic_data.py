"""
Generates a synthetic dataset with the same structure as the real,
merged EICV7 analysis dataset (see src/step1_build_dataset.py), so the
pipeline can be run and tested end to end WITHOUT access to the real,
restricted NISR microdata.

This produces plausible-looking numbers for testing the CODE only.
It is not a substitute for real data and must never be described as a
real result.

Usage:
    python tests/generate_synthetic_data.py --out data/synthetic_dataset.pkl
"""
import argparse
import numpy as np
import pandas as pd


def generate(n_districts: int = 30, n_per_district: int = 500, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_districts):
        u_d = rng.normal(0, 0.2)
        hhsize = rng.poisson(4, n_per_district) + 1
        head_age = rng.normal(45, 12, n_per_district).clip(18, 90)
        head_sex = rng.choice([1, 2], n_per_district)
        electricity_grid = rng.choice([1, 2, 3, 4], n_per_district, p=[0.35, 0.05, 0.05, 0.55])
        n_rooms = rng.poisson(2, n_per_district) + 1
        livestock = rng.poisson(1.5, n_per_district)

        mu = (13.6 - 0.10 * np.log(hhsize) - 0.003 * head_age
              + 0.30 * (electricity_grid == 1) + 0.30 * np.log(n_rooms)
              + 0.08 * np.log1p(livestock) + u_d)
        e = rng.normal(0, 0.48, n_per_district)
        sol_jan = np.exp(mu + e)
        cons1ae = sol_jan * rng.uniform(0.9, 1.3, n_per_district)  # nominal != deflated, by construction
        weight = rng.uniform(150, 300, n_per_district)
        pop_wt = weight * hhsize

        for i in range(n_per_district):
            rows.append({
                "district": 11 + d, "cons1ae": cons1ae[i], "sol_jan": sol_jan[i],
                "weight": weight[i], "pop_wt": pop_wt[i],
                "Poverty_line": 560127, "Extreme_line": 356432,
                "n_rooms": n_rooms[i], "electricity_grid": electricity_grid[i],
                "livestock_index": livestock[i], "head_age": head_age[i], "head_sex": head_sex[i],
                "hhsize": hhsize[i],
            })
    df = pd.DataFrame(rows)
    df["pov_jan"] = (df["sol_jan"] < df["Poverty_line"]).astype(int) * 100
    df["epov_jan"] = (df["sol_jan"] < df["Extreme_line"]).astype(int) * 100
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic_dataset.pkl")
    parser.add_argument("--n-districts", type=int, default=30)
    parser.add_argument("--n-per-district", type=int, default=500)
    args = parser.parse_args()

    df = generate(args.n_districts, args.n_per_district)
    print(f"Generated synthetic dataset: {df.shape}")
    df.to_pickle(args.out)
    print(f"Saved to {args.out}")
