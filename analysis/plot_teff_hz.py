from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from hz.ashz import AlfvenSurfaceHabitableZone
from hz.chz import CircumstellarHabitableZone
from hz.uhz import UltravioletHabitableZoneABG

### READ PLANETARY DATA
df = pd.read_csv("../db/params-step3.csv")
xplot = df["pl_orbsmax"]
yplot = df["st_teff"]



### CREATE HZ CALCULATORS

npts = 10000

# solar luminosity
lumi = np.full(npts, 1.0)

# earth mass
plmass = np.full(npts, 1.0)

# atmospheric transmission
fnuv = np.full(npts, 1.0)

# solar spectral type
lumclass = np.full(npts, 5)

# solar rotation period
prot = np.full(npts, 25.)

# limit to ASHZ calc
a_lo = 1e-2
a_hi = 1e+1
r_lim = np.full(npts, a_hi)

# range of effective temperatures to prob
teff_lo = 3000
teff_hi = 6500
teff = np.linspace(teff_lo, teff_hi, npts)

# check on temp/vk interpolator... wiggly. maybe smooth a bit?
# from hz.ashz._ashz import create_interpolator_temp_vk
# interp = create_interpolator_temp_vk()
# vk = interp(teff, lumclass)
#
# fig, ax = plt.subplots()
# ax.set_ylim(1.1, 7.0)
# ax.plot(teff, vk)
# plt.show()



# Kopparapu+ 14 CHZ
chz = CircumstellarHabitableZone(teff, lumi, plmass)
chz_inner = chz.inner_rad()
chz_outer = chz.outer_rad()

# Spinelli+ 23 UHZ
uhz = UltravioletHabitableZoneABG(teff, fnuv)
uhz_inner = uhz.inner_rad()
uhz_outer = uhz.outer_rad()

# Atkinson+ 24 ASHZ
ashz = AlfvenSurfaceHabitableZone(teff, lumclass, prot, r_lim)
ashz_inner = ashz.inner_rad()
ashz_outer = ashz.outer_rad()

hz_inner = np.max([ashz_inner, chz_inner, uhz_inner], axis=0)
hz_outer = np.min([ashz_outer, chz_outer, uhz_outer], axis=0)

planet_count = np.sum((a_lo < xplot) & (xplot < a_hi) & (teff_lo < yplot) & (yplot < teff_hi))
# planet_hz_count = np.sum()

fig, ax = plt.subplots()
ax.set_xlabel("Orbital Distance (au)", fontsize=14)
ax.set_xscale("log")
ax.set_xlim(1e-2, 1e1)

ax.set_ylabel("Stellar Effective Temperature (K)", fontsize=14)
ax.set_ylim(teff_lo, teff_hi)

ax.scatter(xplot, yplot, color="#AAAAAA", alpha=0.5, s=10, marker=".", label=f"planets ({planet_count})")

ax.plot(ashz_inner, teff, color="C1", lw=2, label=r'ASHZ (P$_{rot}$ = 25d)')
# ax.plot(ashz_outer, teff, color="C1", lw=2)

ax.plot(chz_inner, teff, color="C2", lw=2, label=r'CHZ (M$_{pl}$ = 1M$_\oplus$)')
ax.plot(chz_outer, teff, color="C2", lw=2)

ax.plot(uhz_inner, teff, color="C4", lw=2, label=r'UHZ (f$_{NUV}$ = 0.1)')
ax.plot(uhz_outer, teff, color="C4", lw=2)

ax.fill_betweenx(teff, hz_inner, hz_outer, where=hz_inner < hz_outer, color="#20B0B8", alpha=0.2, hatch="///", lw=0, label="hz")

plt.legend(loc="upper left", fontsize=9)

plt.show()