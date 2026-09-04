"""
Step 3: Poverty indicators under the lognormal model.

Implements the Head Count Ratio and Poverty Gap formulas directly (no
point-prediction shortcut, which discards residual uncertainty):

    a = (log(z) - mu) / sigma_e
    HCR contribution = Phi(a)
    PG contribution  = Phi(a) - exp(mu + sigma_e^2/2)/z * Phi(a - sigma_e)

aggregated to district level using population survey weights.
"""
import numpy as np
import pandas as pd
from scipy import stats


def household_poverty_indicators(mu, sigma_e, poverty_line):
    a = (np.log(poverty_line) - mu) / sigma_e
    hc = stats.norm.cdf(a)
    pg = hc - (np.exp(mu + sigma_e**2 / 2) / poverty_line) * stats.norm.cdf(a - sigma_e)
    return hc, pg


def district_level_indicators(df: pd.DataFrame, mu, sigma_e, weight_col: str,
                                district_col: str, poverty_line: float) -> pd.DataFrame:
    hc, pg = household_poverty_indicators(mu, sigma_e, poverty_line)
    d = df.copy()
    d["_hc"] = hc
    d["_pg"] = pg
    out = d.groupby(district_col).apply(
        lambda g: pd.Series({
            "HCR": np.average(g["_hc"], weights=g[weight_col]) * 100,
            "PG": np.average(g["_pg"], weights=g[weight_col]) * 100,
            "n_households": len(g),
        }),
        include_groups=False
    )
    return out
