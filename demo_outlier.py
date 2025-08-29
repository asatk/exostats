"""
demo_regression.py

Demonstration of identifying outliers in a subset of NASA exoplanet data.

2025.08.29
Anthony Atkinson
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# table to be loaded
fname = "tables-merged/nasa_exo.csv"

# load data set into Pandas
df = pd.read_csv(fname)

# we only care about the magnitude, not direction, of stellar rotation
df["st_rotp"] = np.abs(df["st_rotp"])

# remove 0s in columns where you do not want zeros (useful when taking logs)
nz_cols = ["st_age"]
df[nz_cols] = df[nz_cols].replace([0.0], np.nan)

# subset of planetary systems to investigate (systems with 2 or more rocky pls)
subset = (df["sy_pnum"] > 1) & \
        (0.5 <= df["pl_bmasse"]) & \
        (df["pl_bmasse"] <= 3)

# predictor variable (X, independent variable) -- stellar effective temperature
pred = ["st_teff"]
# response variable -- stellar rotation period
resp = "st_rotp"
# column names for all relevant data
cols = pred + [resp]

# dataframe of desired subset of planets
df_sub = df.loc[subset, cols].dropna()

# define design matrix X = log(teff)
X = np.log10(df_sub[pred]).to_numpy().reshape(-1, 1)
#X = df_sub[pred].to_numpy().reshape(-1, 1)

# define response y = log(Prot)
y = np.log10(df_sub[resp]).to_numpy()
#y = df_sub[resp].to_numpy()

# ordinary least squares model
reg = LinearRegression().fit(X, y)

# calculate leverage
n = len(X)
lev = (X - np.mean(X, axis=0))**2 / np.sum((X - np.mean(X, axis=0))**2) + 1 / n

# identify leverage points
# leverage pt := leverage is greater than 2 x average leverage = 2 x 2 / n
levpt = np.any(lev > 4 / n, axis=1)

# calculate standardized residual
res = reg.predict(X) - y
s = np.sqrt(np.sum((res)**2) / (n - 2))
# TODO change lev back to lev, figure out where multiple reg model is
r = res / s / np.sqrt(1 - lev.ravel())

# identify outliers ("bad" leverage points)
# outlier := leverage point with absolute value of standardized residual >= 2
badlev = (np.abs(r) > 2) & levpt
n_badlev = np.sum(badlev)
print(f"Outliers identified: {n_badlev:d}")

# exclude outliers from previous data set
df_outlr = df_sub.loc[~badlev, cols].dropna()
X_outlr = np.log10(df_outlr[pred]).to_numpy().reshape(-1, 1)
#X_outlr = df_outlr[pred].to_numpy().reshape(-1, 1)
y_outlr = np.log10(df_outlr[resp]).to_numpy()
#y_outlr = df_outlr[resp].to_numpy()

# fit the OLS model on the new data!
reg_outlr = LinearRegression().fit(X_outlr, y_outlr)

# predict rotation periods based on stellar temperatures
xmin = np.min(X)
xmax = np.max(X)
npts = 1000
Xstar = np.linspace(xmin, xmax, npts, endpoint=True).reshape(-1, 1)
ystar = reg.predict(Xstar)
ystar_outlr = reg_outlr.predict(Xstar)

# get power-law coefficient
p = reg.coef_[0]
p_outlr = reg_outlr.coef_[0]

# print planets that are outliers
print(df.loc[df_sub[badlev].index, ["pl_name"] + cols])

# plot data and fits
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.scatter(X[badlev,0], y[badlev], color="red", marker="+",
           label=f"Outliers (N = {n_badlev})")
ax.scatter(X_outlr[:,0], y_outlr[:], color="black", marker="+",
           label=f"Planets (N = {len(df_outlr)})")
ax.plot(Xstar[:,0], ystar, ls="-", label=f"OLS (p = {p:.3f})")
ax.plot(Xstar[:,0], ystar_outlr, ls="-", label=f"OLS -Outliers (p = {p_outlr:.3f})")
ax.set_xlabel(r"$\log_{10}$ Temperature (K)")
ax.set_ylabel(r"$\log_{10}$ Rotation Period (d)")
ax.set_title(r"Removing Outliers in Regression on" + \
        "\nPlanetary Data")
ax.legend()
plt.show()
