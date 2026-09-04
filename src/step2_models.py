"""
Step 2: Model definitions.

  - BHF: standard nested-error unit-level model (GLS/REML mixed model).
  - PM:  adaptive nested-error nonparametric model, i.e. BHF plus an
    identifiable, orthogonalized spline "wiggle" term, fitted as a
    variance-component mixed model (lambda = sigma_g^2/(sigma_g^2+sigma_e^2)
    reparametrizes the smoothing/nonlinearity; lambda=0 recovers BHF exactly).

See docs/theory.pdf for the full derivation (identifiability proof, BLUP
derivation, exact MSE under known variance components, and the correct
boundary-adjusted test for lambda=0).
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_bhf(df: pd.DataFrame, welfare_col: str, covariate_cols: list, district_col: str):
    """Standard BHF model: log(welfare) ~ covariates + (1 | district)."""
    d = df.copy()
    d["log_welfare"] = np.log(d[welfare_col])
    formula = "log_welfare ~ " + " + ".join(covariate_cols)
    model = smf.mixedlm(formula, data=d, groups=d[district_col])
    return model.fit(reml=True)


def orthogonalized_spline_basis(x, X_linear, n_knots: int = 6):
    """
    Truncated-power quadratic spline basis for x, orthogonalized against the
    linear design matrix X_linear (identifiability construction; see
    docs/theory.pdf, Theorem 1). Columns are scaled to unit variance for
    numerical stability of the REML optimizer.
    """
    x = np.asarray(x, dtype=float)
    knots = np.quantile(x, np.linspace(0.05, 0.95, n_knots))
    Z = np.column_stack([np.maximum(x - k, 0) ** 2 for k in knots])

    X = np.asarray(X_linear, dtype=float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    P_X = X @ XtX_inv @ X.T
    Z_tilde = Z - P_X @ Z

    col_sd = Z_tilde.std(axis=0)
    col_sd[col_sd == 0] = 1.0
    Z_tilde = Z_tilde / col_sd
    return Z_tilde, knots


def fit_pm(df: pd.DataFrame, welfare_col: str, covariate_cols: list, district_col: str,
           spline_var: str, n_knots: int = 6):
    """PM model: BHF plus an orthogonalized spline wiggle term on spline_var,
    fitted as a variance component shared globally across all observations."""
    d = df.copy().reset_index(drop=True)
    d["log_welfare"] = np.log(d[welfare_col])

    X_linear = sm.add_constant(d[covariate_cols].values)
    Z_tilde, knots = orthogonalized_spline_basis(d[spline_var].values, X_linear, n_knots=n_knots)

    for k in range(Z_tilde.shape[1]):
        d[f"_spline_{k}"] = Z_tilde[:, k]
    spline_cols = [f"_spline_{k}" for k in range(Z_tilde.shape[1])]
    d["_global"] = "ALL"

    fe_formula = "log_welfare ~ " + " + ".join(covariate_cols)
    vc = {"spline": "0 + " + " + ".join(spline_cols)}
    model = smf.mixedlm(fe_formula, data=d, groups=d[district_col], vc_formula=vc, re_formula="1")
    fit = model.fit(reml=True)
    return fit, knots, spline_cols


def lambda_test(bhf_fit, pm_fit):
    """
    Likelihood-ratio test for H0: sigma_g^2 = 0 (lambda = 0). Uses the
    CORRECT boundary-adjusted null distribution (Self and Liang, 1987): an
    equal mixture of a point mass at zero and a chi-square with 1 degree of
    freedom -- not a naive chi-square test, which would overstate
    significance. See docs/theory.pdf, Proposition 3.
    """
    lr_stat = max(2 * (pm_fit.llf - bhf_fit.llf), 0.0)
    p_value = 0.5 * (1 - stats.chi2.cdf(lr_stat, df=1))
    return lr_stat, p_value
