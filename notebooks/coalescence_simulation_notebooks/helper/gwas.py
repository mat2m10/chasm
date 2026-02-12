import numpy as np
from scipy import stats
import statsmodels.api as sm
import pandas as pd

def fit_line(x_, y_, covs=None):
    x_ = np.asarray(x_).ravel()
    y_ = np.asarray(y_).ravel()

    # Original behavior if no covariates
    if covs is None:
        slope, intercept, r, p, se = stats.linregress(x_, y_)
        neglogp = -np.log10(p) if p > 0 else np.inf
        return slope, intercept, neglogp

    # With covariates: OLS on [x, covs] + intercept
    covs = np.asarray(covs)
    if covs.ndim == 1:
        covs = covs.reshape(-1, 1)

    X = np.column_stack([x_, covs])
    X = sm.add_constant(X, has_constant="add")  # adds intercept

    model = sm.OLS(y_, X, missing="drop").fit()

    slope = float(model.params[1])      # coefficient for x_
    intercept = float(model.params[0])  # intercept
    p = float(model.pvalues[1])         # p-value for x_ coefficient

    neglogp = -np.log10(p) if p > 0 else np.inf
    return slope, intercept, neglogp