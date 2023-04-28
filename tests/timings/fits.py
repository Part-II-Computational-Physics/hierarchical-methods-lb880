import numpy as np
from scipy.optimize import curve_fit

def fmm_terms(FILE_PATH):
    with np.load(FILE_PATH) as data:
        averages = data['fmm_averages']
        terms_vals = data['terms_vals']

    def quadratic(x,a,b,c):
        return a*x**2 + b*x + c

    params, covar = curve_fit(quadratic, terms_vals, averages)
    fit = quadratic(terms_vals, *params)
    return params, fit
