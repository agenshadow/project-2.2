import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

file: str = "metingen.csv"

df = pd.read_csv(file, names=["hoek", "N"])

def model(x, n):
    nlucht: float = 1.00029
    d: float = 2e-3  # 2mm
    lamda: float = 532e-9  # 532nm

    theta2 = np.arcsin(np.sin(x)*nlucht/n)

    N = 2*d/lamda*(((n - nlucht*np.tan(x)*np.cos(x)*np.sin(theta2))/np.cos(theta2)) + np.tan(x)*np.sin(x)-n-nlucht-nlucht/np.cos(x))

    return N

param, param_cov = curve_fit(model, df["hoek"], df["N"])

print(param)