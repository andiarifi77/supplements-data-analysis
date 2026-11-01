"""Dy modelet e dokumentimit: regresioni i thjeshtë dhe i shumëfishtë."""

import pandas as pd
import statsmodels.api as sm

TARGET = "log_popularity"
MODELS = {
    "M1_simple": ["log_citations"],
    "M2_multiple": ["log_citations", "log_mean_studies"],
}


def fit_models(data):
    models, comparisons, coefficients = {}, [], []
    for name, predictors in MODELS.items():
        x = sm.add_constant(data[predictors], has_constant="add")
        # OLS minimizon shumën e katrorëve të mbetjeve.
        model = sm.OLS(data[TARGET], x).fit()
        models[name] = model
        comparisons.append({"model": name, "n": int(model.nobs), "R2": model.rsquared})
        for term, value in model.params.items():
            coefficients.append({"model": name, "term": term, "coefficient": value})

    main = models["M2_multiple"]
    residuals = data[["supplement", TARGET]].copy()
    residuals["fitted"] = main.fittedvalues
    # Mbetja = vlera reale - vlera e parashikuar, në shkallën logaritmike.
    residuals["residual"] = data[TARGET] - main.fittedvalues
    return models, pd.DataFrame(comparisons), pd.DataFrame(coefficients), residuals
