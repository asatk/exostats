from astropy import units as u
from astropy import constants as c
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from funcs import tauc_w18_sm
from funcs import ra_s03
from funcs import chz_inner_k14
from funcs import chz_outer_k14


koi = pd.read_csv("../db/kepler/koi_master.csv")

# Kopparapu+ 14 CHZ
stlum = 4 * np.pi * (koi["koi_srad"].to_numpy() * u.R_sun) ** 2 * c.sigma_sb * (koi["koi_steff"].to_numpy() * u.K) ** 4
stlum = stlum.to(u.L_sun).value
koi["stlum"] = stlum
koi["chz_inner"] = chz_inner_k14(koi["koi_steff"], koi["stlum"])
koi["chz_outer"] = chz_outer_k14(koi["koi_steff"], koi["stlum"])


# Atkinson+ 24 ASHZ
koi["tauc"] = tauc_w18_sm(koi["koi_smass"].to_numpy())
koi["Ro"] = koi["Prot"] / koi["tauc"]
koi["RA"] = ra_s03(koi["Ro"])
koi["rp"] = koi["koi_sma"] * (1 - koi["koi_eccen"])
koi["ASHC"] = koi["rp"] / koi["RA"]

print(koi["ASHC"].describe())

koi.to_csv("../db/analysis/koi.csv")

is_cand = koi["koi_disposition"] == "CANDIDATE"
is_conf = koi["koi_disposition"] == "CONFIRMED"

koi_cand = koi.loc[is_cand]
koi_conf = koi.loc[is_conf]

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(koi_conf["Ro"], koi_conf["ASHC"], s=5, alpha=0.5, label="Kepler Planets")
ax.scatter(koi_cand["Ro"], koi_cand["ASHC"], s=5, alpha=0.5, label="Kepler Candidates")
ax.set_xlabel("Rossby Number Ro")
ax.set_xlim(0.0, 5.0)
ax.set_ylabel("ASHC")
ax.set_ylim(7e-2, 4e1)
ax.set_yscale("log")
ax.set_title("Kepler Objects of Interest -- Stellar Activity Habitability")
ax.legend()
plt.show()
