"""
Step 1: Build the analysis dataset.

Merges NISR EICV7 files (poverty file, household file, person file) into a
single household-level analysis dataset. Real data files are NOT included in
this repository (see README) -- point DATA_DIR at your local extraction of
the NISR microdata download.

Usage:
    python -m src.step1_build_dataset --data-dir /path/to/Cross_Section --out data/analysis_dataset.pkl
"""
import argparse
import pandas as pd


def build_dataset(data_dir: str) -> pd.DataFrame:
    # --- Poverty file: welfare, weights, poverty lines, district ---
    pov = pd.read_stata(f"{data_dir}/CS_EICV7_poverty_file.dta", convert_categoricals=False)
    pov = pov[["hhid", "district", "province", "ur", "cons1ae", "sol_jan", "weight", "pop_wt",
               "Poverty_line", "Extreme_line", "pov_jan", "epov_jan"]].copy()
    # NOTE: `sol_jan` is NISR's deflated welfare aggregate actually used for
    # official poverty classification; `cons1ae` is nominal/undeflated and
    # should NOT be used as the welfare outcome (verified: sol_jan < Poverty_line
    # matches pov_jan for 100% of households in the real EICV7 file).

    # --- Household file: housing / asset covariates ---
    hh_raw = pd.read_stata(f"{data_dir}/CS_S01_S5_S7_Household.dta", convert_categoricals=False)
    hh_cov = hh_raw[["hhid", "s5aq3", "s5cq14", "s5cq25",
                      "s7aq4a", "s7aq4b", "s7aq4c", "s7aq4d", "s7aq4e"]].copy()
    hh_cov = hh_cov.rename(columns={
        "s5aq3": "n_rooms", "s5cq14": "electricity_grid", "s5cq25": "toilet_type",
        "s7aq4a": "n_cattle", "s7aq4b": "n_goats", "s7aq4c": "n_sheep",
        "s7aq4d": "n_pigs", "s7aq4e": "n_poultry",
    })
    for c in ["n_cattle", "n_goats", "n_sheep", "n_pigs", "n_poultry"]:
        hh_cov[c] = hh_cov[c].fillna(0)
    hh_cov["livestock_index"] = hh_cov[["n_cattle", "n_goats", "n_sheep", "n_pigs", "n_poultry"]].sum(axis=1)

    # --- Person file, filtered to household heads (s1q2 == 1) ---
    person = pd.read_stata(f"{data_dir}/CS_S0_S1_S2_S3_S4_S6A_S6B_S6C_Person.dta",
                            convert_categoricals=False)
    heads = person[person["s1q2"] == 1].copy()
    head_cov = heads[["hhid", "s1q1", "s1q3y", "s4aq2"]].rename(columns={
        "s1q1": "head_sex", "s1q3y": "head_age", "s4aq2": "head_educ_level",
    })
    hh_size = person.groupby("hhid").size().rename("hhsize").reset_index()

    df = pov.merge(hh_cov, on="hhid", how="left") \
            .merge(head_cov, on="hhid", how="left") \
            .merge(hh_size, on="hhid", how="left")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to extracted Cross_Section folder")
    parser.add_argument("--out", default="data/analysis_dataset.pkl")
    args = parser.parse_args()

    df = build_dataset(args.data_dir)
    print(f"Built dataset: {df.shape}")
    df.to_pickle(args.out)
    print(f"Saved to {args.out}")
