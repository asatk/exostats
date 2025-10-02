"""
demo_regression.py

Demonstration of linear regression on a subset of NASA exoplanet data with
different regression models.

2025.07.02
Anthony Atkinson
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

# table to be loaded
fname = "tables-merged/nasa_exo.csv"

# load data set into Pandas
df = pd.read_csv(fname)

# subset of planetary systems to investigate
subset = (df["sy_pnum"] == 1) & (0.25 <= df["pl_bmassj"]) & \
        (df["pl_bmassj"] <= 13)
# columns of relevant parameters for fit
cols = ["st_age", "st_rotp"]
# dataframe of single-giant-planet systems w/ age and stellar rotation period
df_sgl = df.loc[subset, cols].dropna()

# test Skumanich's Law for single-giant-planet systems: Prot ~ age^0.5
# define design matrix X = log(age)
X = np.log10(df_sgl["st_age"].to_numpy().reshape(-1, 1))
# define response y = log(Prot)
y = np.log10(np.fabs(df_sgl["st_rotp"].to_numpy()))

# regularization strengths for L1 and L2 penalties in each model
alpha_ridge = 10.0
alpha_lasso = 0.01
alpha_enet = 0.1
l1_ratio = 0.05

# try different regression models
reg_OLS = LinearRegression().fit(X, y)
reg_ridge = Ridge(alpha=alpha_ridge).fit(X, y)
reg_lasso = Lasso(alpha=alpha_lasso).fit(X, y)
reg_enet = ElasticNet(alpha=alpha_enet, l1_ratio=l1_ratio).fit(X, y)

# predict rotation periods based on stellar ages
xmin = np.min(X)
xmax = np.max(X)
npts = 1000
Xstar = np.linspace(xmin, xmax, npts, endpoint=True).reshape(-1, 1)
ystar_OLS = reg_OLS.predict(Xstar)
ystar_ridge = reg_ridge.predict(Xstar)
ystar_lasso = reg_lasso.predict(Xstar)
ystar_enet = reg_enet.predict(Xstar)

# get power-law coefficient
p_OLS = reg_OLS.coef_[0]
p_ridge = reg_ridge.coef_[0]
p_lasso = reg_lasso.coef_[0]
p_enet = reg_enet.coef_[0]

# plot data and fits
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.scatter(X[:,0], y, color="black", marker="+", label="Single HJ Systems " + \
           f"(N = {len(df_sgl)})")
ax.plot(Xstar[:,0], ystar_OLS, ls="-", label=f"OLS (p = {p_OLS:.3f})")
ax.plot(Xstar[:,0], ystar_ridge, ls="-", label=f"Ridge (p = {p_ridge:.3f})")
ax.plot(Xstar[:,0], ystar_lasso, ls="-", label=f"Lasso (p = {p_lasso:.3f})")
ax.plot(Xstar[:,0], ystar_enet, ls="-", label=f"Elastic Net (p = {p_enet:.3f})")
ax.set_xlabel(r"$\log_{10}$ Age (Myr)")
ax.set_ylabel(r"$\log_{10}$ Stellar Rotation Period (d)")
ax.set_title(r"Testing Skumanich's Law ($P_{rot}\propto t^{0.5}$)" + \
        "\nin Single Hot Jupiter Planetary Systems")
ax.legend()
plt.show()
