import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
file = r"MeetData/temp_metingen.csv"

df = pd.read_csv(file)
df["Ngem"] = (df["N1"] + df["N2"] + df["N3"] + df["N4"] + df["N5"])/5
df["stdv"] = (((df["N1"] - df["Ngem"])**2 + (df["N2"] - df["Ngem"])**2 + (df["N3"] - df["Ngem"])**2 + (df["N4"] - df["Ngem"])**2 + (df["N5"] - df["Ngem"])**2) / 4)**(1/2)

def liniear_model(x, a, b):
    N = a*x+b
    return N

params, pcov = curve_fit(liniear_model, df["delta T"], df["Ngem"],sigma=df["stdv"],absolute_sigma=True)


T = np.linspace(21, 50, 100)


d = 1e-2
lamda = 532e-9
a = params[0]*lamda/(2*d)

plt.plot(T, liniear_model(T, a,1.36), c="deepskyblue")

plt.xlabel("T [°C]")
plt.ylabel("n [-]")
plt.title("brekingsindex tegen de tempratuur")
plt.show()