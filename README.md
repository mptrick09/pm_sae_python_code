# Python Code — Small Area Estimation Application to Rwanda's EICV7

This package contains only the Python analysis code used in the paper *"Small Area Estimation and
the Sources of Regional Poverty Disparities in Rwanda: An Application to EICV7."* It excludes the
R simulation study, the paper itself, and theoretical derivations — see the full project repository
for those.

## What is, and is not, original here

This code implements established statistical methodology (Battese, Harter, and Fuller, 1988;
Opsomer et al., 2008; Self and Liang, 1987; Crainiceanu and Ruppert, 2004) applied to a real
dataset. The model and hypothesis test are not novel; the application, the data correction
documented in `src/step1_build_dataset.py`, and the analysis built from it are.

## Files

| File | Purpose |
|---|---|
| `src/step1_build_dataset.py` | Merges NISR's EICV7 poverty, household, and person files into one analysis dataset. Documents why `sol_jan`, not `cons1ae`, is the correct welfare variable. |
| `src/step2_models.py` | Fits the standard model (BHF) and its nonparametric extension (PM) via `statsmodels`; implements the boundary-adjusted significance test. |
| `src/step3_poverty_indicators.py` | Computes household- and district-level Head Count Ratio and Poverty Gap from the fitted model, using the exact lognormal formulas. |
| `src/step4_run_analysis.py` | Runs steps 1-3 end to end from the command line. |
| `tests/generate_synthetic_data.py` | Generates a synthetic dataset with the same structure as real EICV7 data, so the pipeline can be run and verified without access to the real (restricted) microdata. |
| `notebooks/01_walkthrough.ipynb` | The same pipeline, broken into cells, for interactive use in Jupyter or VS Code. |

## No real data is included

EICV7 is restricted microdata (National Institute of Statistics of Rwanda). None of it is included
here. To run on real data, obtain it from https://microdata.statistics.gov.rw and point
`step1_build_dataset.py` at your local copy. To test the code without real data:

```
pip install -r requirements.txt
python tests/generate_synthetic_data.py --out data/synthetic_dataset.pkl
python -m src.step4_run_analysis --data data/synthetic_dataset.pkl --welfare-col sol_jan
```

## Important data note

Use `sol_jan` (deflated welfare), not `cons1ae` (nominal). Using the wrong variable changes the
implied national poverty rate by more than 10 percentage points. See `src/step1_build_dataset.py`
for the verification.
