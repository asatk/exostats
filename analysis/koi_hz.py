from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


w18_sm_c0 = 2.33
w18_sm_c1 = -1.50
w18_sm_c2 = 0.31

def tauc_w18_sm(mass: np.ndarray):

    mass_np = mass.to_numpy()

    oob = (0.08 > mass_np) | (mass_np > 1.36)
    mass_np[oob] = np.nan

    log_tauc = w18_sm_c0 + w18_sm_c1 * mass_np + w18_sm_c2 * mass_np ** 2

    return 10. ** log_tauc

r_sun = 6.957e8
au = 1.496e11

s = -1.38
r = -0.16
Ro_sun = 1.85
ra_sun = 20 * r_sun / au

def ra_s03(Ro: np.ndarray):
    return ra_sun * np.real(np.power(Ro / Ro_sun, s * r))

koi = pd.read_csv("../db/kepler/koi_master.csv")


koi["tauc"] = tauc_w18_sm(koi["koi_smass"])
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
