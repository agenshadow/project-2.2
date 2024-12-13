import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt






file: str = "data/metingen 2.csv"

residuals: list[float] = []

df = pd.read_csv(file, names=["hoek", "N1", "N2", "N3"])
df["Ngem"] = (df["N1"] + df["N2"] + df["N3"])/3
df["stdv"] = (((df["N1"] - df["Ngem"])**2 + (df["N2"] - df["Ngem"])**2 + (df["N3"] - df["Ngem"])**2) / 2)**(1/2)


Nlijst = df["Ngem"]

df["radialen"] = df["hoek"] * (2*np.pi)/360

radialenlijst = df["radialen"].to_numpy()





def model(x, n):
    nlucht = 1.0029
    d: float = 2e-3  # 2mm
    lamda: float = 532e-9  # 532nm

    theta2 = np.arcsin(np.sin(x)*nlucht/n)

    #N = 2*d/lamda*((n - nlucht*(np.sin(theta2))**2)/np.cos(theta2) + np.tan(x)*np.sin(x) - n - nlucht - nlucht/np.cos(x))
    N = 2*d/lamda*(n/np.cos(theta2) + np.tan(x)*np.sin(x)*nlucht - np.tan(theta2)*np.sin(x)*nlucht - (n-nlucht) - nlucht/np.cos(x))
    return N




param, param_cov = curve_fit(model, df["radialen"], df["Ngem"], p0=[1.5], sigma=df["stdv"], absolute_sigma=True)




n = param[0]

print(f"de brekingsindex = {round(n, 5)} met een standaard deviatie van {round(np.sqrt(param_cov[0][0]), 5)}")

for i,value in enumerate(Nlijst):
    overig = value - model(radialenlijst[i], n)
    residuals.append(overig)


print(f"afwijking van litratuur is: {round((1 - n/1.49)*100,2)}%")


fig,ax = plt.subplots(2,1)

ax[0].scatter(radialenlijst, df["Ngem"], label="meetdata")
ax[0].errorbar(radialenlijst, df["Ngem"],yerr=df["stdv"],capsize=4, ls="none")

ax[0].plot(df["radialen"], model(df["radialen"], n), label="fitting op basis van meetdata",c="coral")
ax[0].plot(df["radialen"], model(df["radialen"], 1.49), label="fitting op basis van literatuur waarde", ls="--",c="gray")

ax[0].set_xlabel("hoek in rad")
ax[0].set_ylabel("aantal fringes")
ax[0].set_title("fringes x hoek(rad) grafiek")

ax[0].legend()

ax[1].plot(radialenlijst, residuals)
ax[1].set_xlabel("hoek in rad")
ax[1].set_ylabel("aantal fringes")
ax[1].set_title("residuen")

plt.show()