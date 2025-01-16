# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:00:11 2025

@author: Ben van Merkom
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#Variabelen
nl = 1.00029
dw = 10e-3
lamda = 532e-9
np_ = 1.497

dp = 2e-3

#Metingen
N1 = [3,3,3,3]
N2 = [9,8,9,9]
N3 = [20,21,20,20]
N4 = [26,27,28,28]
N5 = [51,51,52,52]

theta0 = np.radians([1, 2, 3, 4, 5])  #Hoekmetingen in radians
N_meting = [np.mean(N1), np.mean(N2), np.mean(N3), np.mean(N4), np.mean(N5)]  #Gemiddelde fringes

#Functie voor berekening van fringes
def model(nw):
    theta1 = np.arcsin((np.sin(theta0) * nl) / np_)
    theta2 = np.arcsin((np.sin(theta1) * np_) / nw)
    
    N1 = 2*dp/lamda*(np_/np.cos(theta1) + (np.tan(theta0)*np.sin(theta0)*nw - np.tan(theta1)*np.sin(theta0)*nw ) / np.cos(theta2) - (np_-nl) - nl/np.cos(theta0))
    Nw = 2*dw/lamda*(nw /np.cos(theta2) + (np.tan(theta0)*np.sin(theta0)*np_- np.tan(theta2)*np.sin(theta0)*np_) / np.cos(theta1-theta2) - (nw-nl) - nl/np.cos(theta0))
    N2 = 2*dp/lamda*(np_/np.cos(theta1) + (np.tan(theta0)*np.sin(theta0)*nl - np.tan(theta1)*np.sin(theta0)*nl ) - (np_-nl) - nl/np.cos(theta0))
    return Nw-N1-N2


def doel_functie(nw):
    N_model = model(nw)
    fout = np.sum((N_model - N_meting) ** 2)
    return fout

#Optimalisatie
result = minimize(doel_functie, x0=1.333,)  

#Resultaat
nw_optimaal = result.x[0]
print(nw_optimaal)

#Model voorspellingen met nw = 1.333
nw_expected = 1.333
N_model_expected = model(nw_expected)
N_model_metingen = model(nw_optimaal)

#Plot
plt.figure(figsize=(8, 6))
plt.plot(np.degrees(theta0), N_model_expected, label="Model (nw = 1.333)", linestyle='-', marker='o')
plt.plot(np.degrees(theta0), N_model_metingen, label= f"Model metingen (nw = {nw_optimaal:.3f} )", linestyle='-', color="red")
plt.scatter(np.degrees(theta0), N_meting, color='red', label="Meetwaarden", zorder=5)
plt.xlabel("Hoek (graden)")
plt.ylabel("Aantal fringes (N)")
plt.title("Vergelijking van model en meetwaarden")
plt.legend()
plt.grid()
plt.show()