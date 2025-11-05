import astropy
from astropy.modeling import models, fitting
from astropy import units as u
from astropy import constants as c
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

ps = pd.read_csv("../db/nea/ps.csv")
subset = pd.notnull(ps["st_teff"]) & pd.notnull(ps["sy_umag"]) & pd.notnull(ps["sy_dist"]) & \
        (np.log10(ps["st_teff"]) < 4.4)
ps = ps[subset]
teff = ps["st_teff"]
mag_u = ps["sy_umag"]
dist = ps["sy_dist"]
Mag_u = mag_u - 5 * np.log10(dist) + 5
Mag_u_sun = 5.51
lumi_nuv = 10 ** ((Mag_u - Mag_u_sun) / -2.5) * c.L_sun
print(lumi_nuv)

model_init = models.Linear1D(slope=21.12, intercept=48.22)
fitter = fitting.LinearLSQFitter()
model = fitter(model_init, np.log10(teff), np.log10(lumi_nuv))

print(model)
temp_predict = np.linspace(2000, 8000, 1000)
lumi_nuv_predict = 10 ** model(np.log10(temp_predict))

fig, ax = plt.subplots()

ax.scatter(np.log10(teff), np.log10(lumi_nuv))
ax.plot(np.log10(temp_predict), np.log10(lumi_nuv_predict), color="red", ls="--")
ax.set_xlabel("Teff (K)")
ax.set_ylabel(r"L$_{NUV}$")
plt.show()


ms = astropy.table.Table.read("../db/analysis/teff-color-ms.txt", format="ascii", guess=False, delimiter=r"\s")
vk_ms = ms["V-K"]
teff_ms = ms["Te(K)"]
ms.pprint()
spl_init = models.Spline1D()
fitter_spl = fitting.SplineInterpolateFitter()
spl = fitter_spl(spl_init, vk_ms, teff_ms)

vk_test = np.linspace(-0.91, 7.37, 1000)
teff_pred = spl(vk_test)

fig, ax = plt.subplots()
ax.scatter(vk_ms, teff_ms, s=10, color="orange")
ax.plot(vk_test, teff_pred, lw=2, ls="--", c="grey")
ax.set_xlabel("V-K")
ax.set_ylabel("Teff (K)")
plt.show()


fig, ax = plt.subplots()

ax.scatter
ax.set_xlabel("Semi-major Axis (au)")
ax.set_ylabel("Stellar Effective Temperature (K)")

plt.show()
