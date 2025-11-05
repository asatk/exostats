import astropy
from astropy.modeling import models, fitting
from astropy import units as u
from astropy import constants as c
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from funcs import chz_inner_k14
from funcs import chz_outer_k14
from funcs import uhz_inner_s23
from funcs import uhz_outer_s23
from funcs import ashz_inner_a24


# interpolate V-Ks color as a function of temperature
ms = astropy.table.Table.read("../db/analysis/teff-color-ms.txt", format="ascii", guess=False, delimiter=r"\s").to_pandas()
ms.sort_values(by="Te(K)", ascending=True, inplace=True)
subset = (1.1 < ms["V-K"]) & (ms["V-K"] < 7.0)
vk_ms = ms.loc[subset, "V-K"]
teff_ms = ms.loc[subset, "Te(K)"]

# cubic spline
spl_init = models.Spline1D()
fitter_spl = fitting.SplineInterpolateFitter()
spl = fitter_spl(spl_init, teff_ms, vk_ms)

# Spinelli+ 23 EQ 1
lumi_nuv_model = models.Linear1D(slope=21.12, intercept=-48.22)

teff = np.linspace(2850, 6530, 10000)

# Free params
prot = 25.0
f_nuv = 0.1
lumi = 1.0
mass_pl = 1.0

# Kopparapu+ 14 CHZ
chz_in = chz_inner_k14(teff, lumi, mass_pl)
chz_out = chz_outer_k14(teff, lumi)

# Spinelli+ 23 UHZ
lumi_nuv = 10. ** lumi_nuv_model(np.log10(teff))
uhz_in = uhz_inner_s23(teff, f_nuv, lumi_nuv)
uhz_out = uhz_outer_s23(teff, f_nuv, lumi_nuv)

# Atkinson+ 24 ASHZ
vk = spl(teff)
ashz_in = ashz_inner_a24(teff, prot, vk)

# Print Sun's HZ
i = np.argmin(np.fabs(5776 - teff))
print(f"Temp: {teff[i]:.1f} K")
print(f"CHZ: {chz_in[i]:.2f} -- {chz_out[i]:.2f} au")
print(f"UHZ: {uhz_in[i]:.2f} -- {uhz_out[i]:.2f} au")
print(f"ASHZ: {ashz_in[i]:.2f} -- [inf] au")

# Load planets and candidates
ps_ = pd.read_csv("../db/nea/ps.csv")
koi_ = pd.read_csv("../db/nea/koi.csv")
k2_ = pd.read_csv("../db/nea/k2.csv")
toi_ = pd.read_csv("../db/nea/toi.csv")

ps = ps_[["pl_name", "pl_orbsmax", "st_teff", "sy_umag", "sy_dist", "pl_bmasse", "st_rotp"]]
ps["chz_in"] = chz_inner_k14(ps["st_teff"], ps["st_lum"], ps["pl_bmasse"])

fig, ax = plt.subplots()

xmax = 1e1

ax.fill_betweenx(teff, chz_in, chz_out, color="green", alpha=0.5, label="CHZ", zorder=3)
ax.fill_betweenx(teff, uhz_in, uhz_out, color="violet", alpha=0.5, label="UHZ", zorder=2)
ax.fill_betweenx(teff, ashz_in, xmax, color="C1", alpha=0.2, label="ASHZ", zorder=1)
"""
ax.plot(chz_in, teff, color="green", zorder=3, lw=1)
ax.plot(chz_out, teff, color="green", zorder=3, lw=1)
ax.plot(uhz_in, teff, color="violet", zorder=2, lw=1)
ax.plot(uhz_out, teff, color="violet", zorder=2, lw=1)
ax.plot(ashz_in, teff, color="C1", zorder=1, lw=1)
"""
#ax.scatter(ps["pl_orbsmax"], ps["st_teff"], s=5, color="grey", alpha=0.5, label="Planet", zorder=0)
ax.set_xlabel(r"a (au)", fontsize=14)
ax.set_xlim(1e-2, xmax)
ax.set_xscale("log")
ax.set_ylabel(r"T$_{eff}$ (K)", fontsize=14)
ax.set_ylim(3000, 6500)
ax.set_title("Theoretical Habitable Zones:\n" + \
        rf"M$_{{pl}}$ = {mass_pl:.1f} M$_{{\oplus}}, $P$_{{rot}}$ = {prot:.1f} d, f$_{{NUV}}$ = {f_nuv:.1f}", fontsize=18)
ax.legend(loc=2)

fig.tight_layout()

plt.show()
fig.savefig(f"hz_M{mass_pl:.1f}_P{prot:.1f}_F{f_nuv:.1f}.png")


