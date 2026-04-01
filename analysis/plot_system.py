from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from hz.ashz import AlfvenSurfaceHabitableZone
from hz.chz import CircumstellarHabitableZone
from hz.uhz import UltravioletHabitableZone



pd.set_option("display.max_rows", 100)

# Load system data into memory
df = pd.read_csv("../db/alfven_data.csv")
df.set_index("pl-name", inplace=True)

# plname = "TRAPPIST-1 d"
plname = "TOI-700 d"
# plname = "55 Cnc e"

hostname_pl = df.loc[plname, "hostname"]
letter_pl = df.loc[plname, "pl-letter"]
cols = ["hostname", "st-lum", "e_st_lum", "st-teff", "Prot", "e_Prot", "pl-bmasse", "e_pl_bmasse", "VK-color", "e_VK_color", "pl-orbsmax", "pl-orbeccen", "rperi"]
cond_pl =  df.index == plname
cond_system = (df["hostname"] == hostname_pl) & ~cond_pl
pd.concat([df.loc[[plname], cols], df.loc[cond_system, cols]])

lumi = df["st-lum"].to_numpy()
e_lumi = df["e_st_lum"].to_numpy()
teff = df["st-teff"].to_numpy()
e_teff = 0.1 * teff
prot = df["Prot"].to_numpy()
e_prot = df["e_Prot"].to_numpy()
plmass = df["pl-bmasse"].to_numpy()
e_plmass = df["e_pl_bmasse"].to_numpy()
rperi = df["rperi"].to_numpy()
e_rperi = df["e_rperi"].to_numpy()
vk = df["VK-color"].to_numpy()
e_vk = df["e_VK_color"].to_numpy()
eccen = df["pl-orbeccen"].to_numpy()
e_eccen = df["e_pl_orbeccen"].to_numpy()
smax = df["pl-orbsmax"].to_numpy()
e_smax = df["e_pl_orbsmax"].to_numpy()


lumi_pl = lumi[cond_pl]
e_lumi_pl = e_lumi[cond_pl]
teff_pl = teff[cond_pl]
e_teff_pl = e_teff[cond_pl]
prot_pl = prot[cond_pl]
e_prot_pl = e_prot[cond_pl]
plmass_pl = plmass[cond_pl]
e_plmass_pl = e_plmass[cond_pl]
rperi_pl = rperi[cond_pl]
e_rperi_pl = e_rperi[cond_pl]
vk_pl = vk[cond_pl]
e_vk_pl = e_vk[cond_pl]
eccen_pl = eccen[cond_pl]
e_eccen_pl = e_eccen[cond_pl]
smax_pl = smax[cond_pl]
e_smax_pl = e_smax[cond_pl]
fnuv_pl = np.array([0.9])

def ellipse(a, e, th):
    return a * (1 - e**2) / (1 - e * np.cos(th))

# Kopparapu+ 14 CHZ
chz = CircumstellarHabitableZone()
chz_lims = chz.limits(teff_pl, 10**lumi_pl, plmass_pl)

chz_in_pl = chz_lims[0]
chz_out_pl = chz_lims[1]

# Spinelli+ 23 UHZ
uhz = UltravioletHabitableZone()
uhz_lims = uhz.limits(teff_pl, fnuv_pl)

uhz_in_pl = uhz_lims[0]
uhz_out_pl = uhz_lims[1]

# Atkinson+ 24 ASHZ
ashz = AlfvenSurfaceHabitableZone()
ashz_lims = ashz.limits(teff_pl, lumclass_pl, prot)

ashz_in_pl = ashz_lims[0]
ashz_out_pl = ashz_lims[1]



smax_max = np.max(smax[cond_system | cond_pl])
smix_min = np.min(smax[cond_system | cond_pl] * np.sqrt((1 - eccen[cond_system | cond_pl]**2)))
# cond_rocky_system = (df["hostname"] == hostname_pl) & ~cond_pl & ((df["pl-bmasse"] < 10) | (df["pl-orbsmax"] < smax_max
#                                                                                             ))
cond_rocky_system = cond_system
# ashz_out_pl = smax_max * 1.15
ashz_out_pl = 0.35

# TODO add warning for HZs that exceed limits

# TODO add warning for planet that is very far out from rest of system and very massive
print(df.loc["TOI-700 d"])

info_str = \
f"""CHZ = ({chz_in_pl}, {chz_out_pl})
UHZ = ({uhz_in_pl}, {uhz_out_pl})
ASHZ = ({ashz_in_pl})
System orbits = ({smix_min}, {smax_max})
"""
print(info_str)

# plotting coordinates for circular HZs
npts = 1000
t = np.linspace(0, 2*np.pi, npts, endpoint=True)
chz_in_polar = np.repeat(chz_in_pl, repeats=npts)
chz_out_polar = np.repeat(chz_out_pl, repeats=npts)
uhz_in_polar = np.repeat(uhz_in_pl, repeats=npts)
uhz_out_polar = np.repeat(uhz_out_pl, repeats=npts)
ashz_in_polar = np.repeat(ashz_in_pl, repeats=npts)
ashz_out_polar = np.repeat(ashz_out_pl, repeats=npts)

# canvas for plot elements
fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True), dpi=150.0)
cmap = plt.get_cmap("tab10")

# host marker and label
ax.scatter(0, 0, s=100, marker="*", color="gold", lw=0)
ax.text(np.pi/2, ashz_out_pl * 0.1, hostname_pl, color="dimgray", ha="center", va="top")

# target planet orbit and label
orb_pl = ellipse(smax_pl, eccen_pl, t)
p1 = ax.plot(t, orb_pl, color="darkgray", lw=0.5, alpha=1, zorder=1)
ax.scatter(np.pi, rperi_pl, zorder=2)
ax.text(np.pi*.99, rperi_pl*.995, letter_pl, color="dimgray", ha="left", va="bottom")

# system's rocky planets' orbits and labels
smax_sys = smax[cond_rocky_system]
eccen_sys = eccen[cond_rocky_system]
rperi_sys = rperi[cond_rocky_system]
plnames_sys = df.loc[cond_rocky_system, "pl-letter"]
for a, e, rp, name in zip(smax_sys, eccen_sys, rperi_sys, plnames_sys):
    orb_sys = ellipse(a, e, t)
    p2 = ax.plot(t, orb_sys, color="darkgray", lw=0.5, alpha=1, zorder=1)
    ax.scatter(np.pi, rp, zorder=2)
    ax.text(np.pi*.99, rp*.995, name, color="dimgray", ha="left", va="bottom")

# HZ annuli
ax.fill_between(t, chz_in_polar, chz_out_polar, color=cmap(2), alpha=0.2, label="CHZ", edgecolor=None, zorder=0)
ax.fill_between(t, uhz_in_polar, uhz_out_polar, color=cmap(4), alpha=0.2, label="UHZ", edgecolor=None, zorder=0)
ax.fill_between(t, ashz_in_polar, ashz_out_polar, color=cmap(1), alpha=0.2, label="ASHZ", edgecolor=None, zorder=0)

# HZ labels
ax.text(3*np.pi/2, uhz_in_pl*1.1, "U HZ", ha="center", va="top", color="dimgray", fontsize=8)
ax.text(3*np.pi/2, ashz_in_pl*1.0, "AS+U HZ", ha="center", va="top", color="dimgray", fontsize=8)
ax.text(3*np.pi/2, chz_in_pl*1.0, "C+AS+U HZ", ha="center", va="top", color="dimgray", fontsize=8)
ax.text(3*np.pi/2, chz_out_pl*0.99, "C+AS HZ", ha="center", va="bottom", color="dimgray", fontsize=8)
ax.text(3*np.pi/2, ashz_out_pl*0.99, "AS HZ", ha="center", va="bottom", color="dimgray", fontsize=8)

# distance labels
ax.text(-3*np.pi/4, ashz_in_pl, f"{ashz_in_pl[0]:.2f} au", color="dimgray", fontsize=8, ha="center", va="top")
ax.text(-3*np.pi/4, chz_in_pl, f"{chz_in_pl[0]:.2f} au", color="dimgray", fontsize=8, ha="center", va="top")
ax.text(-3*np.pi/4, chz_out_pl, f"{chz_out_pl[0]:.2f} au", color="dimgray", fontsize=8, ha="center", va="top")

# remove polar plot tick marks except at distinct HZ locations
ax.set_xticks([])
ax.set_ylim(0, ashz_out_pl)
locs = np.ravel([ashz_in_pl, chz_in_pl, chz_out_pl])
ax.set_yticks([])
ax.set_title(f"{hostname_pl} System and Habitable Zones")

# remove polar plot gridlines
# fig.patch.set_visible(False)
ax.patch.set_visible(False)
ax.spines["polar"].set_visible(False)

fig.tight_layout()
plt.legend()
plt.show()
plt.savefig(f"{hostname_pl}_system.png", format="png", transparent=True)
