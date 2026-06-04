import argparse
import argparse as ap
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.path import Path
import numpy as np
import pandas as pd
import sys

from hz.ashz import AlfvenSurfaceHabitableZone
from hz.chz import CircumstellarHabitableZone
from hz.uhz import UltravioletHabitableZoneABG


F_NUV = 0.1
LUM_CLASS = 5



parser = ap.ArgumentParser(prog="plot_planet_hz.py",
                           usage="%(prog)s <plname> [options]",
                           description="",
                           formatter_class=ap.ArgumentDefaultsHelpFormatter)

parser.add_argument("plname")
# parser.add_argument("-p", "--plot_lim", type=float)
parser.add_argument("--fnuv", default=F_NUV, type=float, help="transmission of NUV emission through planetary atmosphere.")
parser.add_argument("-f", "--files", nargs="+", default=["../db/solar_system.csv"], help=".csv files containing planetary system data.")
parser.add_argument("-c", "--conditions", default="df['pl-name'] != ''", help="conditions filtering planets. Use 'df' to reference the dataframe; numpy functions are available.")

args = vars(parser.parse_args())



plname = args["plname"]
files = args["files"]
cond_s = args["conditions"]
fnuv_val = args["fnuv"]

# read data tables
df = pd.read_csv(files[0])
if "fictional" not in df.columns:
    df["fictional"] = False
else:
    df["fictional"] = df["fictional"].astype(bool)

if "plot-label" not in df.columns:
    df["plot-label"] = ""


for filename_i in files[1:]:
    df_i = pd.read_csv(filename_i)

    if "fictional" not in df_i.columns:
        df_i["fictional"] = False
    else:
        df_i["fictional"] = df_i["fictional"].astype(bool)

    if "plot-label" not in df.columns:
        df["plot-label"] = ""

    df = pd.merge(df, df_i, how="outer")


# determine filtering conditions
cond = eval(cond_s)



# r_lim = np.max(df["pl-orbsmax"] * (1 - df["pl-orbeccen"]))
r_lim = 100.
# cond = (np.abs(df["pl-orbsmax"]) < 5.0) & (df["fictional"] == False)
# cond = (df["fictional"] == False)
df = df.loc[cond].reset_index()



lumi = df["st-lum"].to_numpy()
# e_lumi = df["e_st_lum"].to_numpy()
teff = df["st-teff"].to_numpy()
# e_teff = 0.1 * teff
prot = df["Prot"].to_numpy()
# e_prot = df["e_Prot"].to_numpy()
plmass = df["pl-bmasse"].to_numpy()
# e_plmass = df["e_pl_bmasse"].to_numpy()
rperi = (1 - df["pl-orbeccen"].to_numpy()) * df["pl-orbsmax"].to_numpy()
# e_rperi = df["e_rperi"].to_numpy()
rapo = (1 + df["pl-orbeccen"].to_numpy()) * df["pl-orbsmax"].to_numpy()
# e_rapo
eccen = df["pl-orbeccen"].to_numpy()
# e_eccen = df["e_pl_orbeccen"].to_numpy()
smax = df["pl-orbsmax"].to_numpy()
# e_smax = df["e_pl_orbsmax"].to_numpy()



fnuv = np.repeat([fnuv_val], repeats=len(df))
# lumclass = df["lumclass"].to_numpy()
lumclass = np.repeat([LUM_CLASS], repeats=len(df))



def ellipse(a, e, theta):
    r = a * (1 - e**2) / (1 - e * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=0)

def ellipse_polar(a, e, npts: int):
    theta = np.linspace(0, 2 * np.pi, npts, endpoint=True)
    r = a * (1 - e**2) / (1 - e * np.cos(theta))
    return np.stack([theta, r], axis=0)



npts = 1000
theta = np.full(npts, np.pi / 2)
phi = np.linspace(0, 2 * np.pi, npts, endpoint=True)

# Kopparapu+ 14 CHZ
chz = CircumstellarHabitableZone(teff, 10**lumi, plmass)
# chz_zone = chz.zone(theta, phi)

# Spinelli+ 23 UHZ
uhz = UltravioletHabitableZoneABG(teff, fnuv)
# uhz_zone = uhz.zone(theta, phi)

# Atkinson+ 24 ASHZ
# r_lim = np.max(rapo)
# print(f"Max radius: {r_lim:.03f} au")
ashz = AlfvenSurfaceHabitableZone(teff, lumclass, prot, r_lim)
# ashz_zone = ashz.zone(theta, phi)



chz_lo, chz_hi = chz.zone(theta, phi)
uhz_lo, uhz_hi = uhz.zone(theta, phi)
ashz_lo, ashz_hi = ashz.zone(theta, phi)


ind_plname = df["pl-name"].isin([plname])

if not np.any(ind_plname):
    print(f"{plname} not found in database.")
    exit(1)

i = np.where(ind_plname)[0][0]
hostname = df.at[i, "hostname"]
print(f"Characterizing the habitability of planet: {df.loc[i,'pl-name']}")
chz_in = np.full_like(phi, chz.inner_rad()[i])
chz_out = np.full_like(phi, chz.outer_rad()[i])
uhz_in = np.full_like(phi, uhz.inner_rad()[i])
uhz_out = np.full_like(phi, uhz.outer_rad()[i])
ashz_in = np.full_like(phi, ashz.inner_rad()[i])
ashz_out = np.full_like(phi, ashz.outer_rad()[i])
# ashz_out = np.full_like(phi, np.inf)

hz_in_abg = np.max([chz_in, uhz_in, ashz_in], axis=0)
hz_out_abg = np.min([chz_out, uhz_out, ashz_out], axis=0)
hz_in = np.max([chz_in, uhz_in, ashz_in], axis=0)
hz_out = np.min([chz_out, ashz_out], axis=0)



rows_system = np.where(df["hostname"].isin([hostname]))[0]
df_system = df.loc[rows_system]



# plot variables
text_frac = 0.025
ruler_frac = 0.125
arrow_frac = 0.25
plot_lim = rapo[i] * 1.1

th_ruler_start = -np.pi/2
th_ruler_end = np.atan(ruler_frac * plot_lim / rperi[i]) - np.pi
th_ruler_text = np.atan((ruler_frac + text_frac) * plot_lim / (rperi[i] / 2)) - np.pi

r_ruler_start = ruler_frac * plot_lim
r_ruler_end = ruler_frac * plot_lim / np.sin(-th_ruler_end)
r_ruler_text = (ruler_frac + text_frac) * plot_lim / np.sin(-th_ruler_text)

th_ruler = np.linspace(th_ruler_start, th_ruler_end, npts, endpoint=True)
r_ruler = ruler_frac * plot_lim / np.sin(-th_ruler)
vertices_ruler = np.stack((th_ruler, r_ruler), axis=1)
path_ruler = Path(vertices_ruler)

# create plot
fig, ax = plt.subplots(subplot_kw=dict(projection="polar"), figsize=(6,6))

# host star
if hostname != "Sol":
    # ax.set_title(f"{hostname} System")

    ax.scatter(0, 0, zorder=5, color="black", marker=".", edgecolors="black", lw=0.0, s=25)
    ax.text(np.pi/2, text_frac * plot_lim, hostname, fontsize=16, ha="center", va="bottom", zorder=5)
else:
    # ax.set_title(f"Solar System")
    ax.text(0, 0, "☉", fontsize=16, ha="center", va="center", zorder=5)


# maximal HZs
# ax.fill_between(phi, hz_in, hz_out, color="#CCCCCC", alpha=0.5, label="hz", hatch="\\\\\\\\", lw=0, where=hz_in < hz_out, zorder=0)
ax.fill_between(phi, hz_in_abg, hz_out_abg, color="#BBBBBB", alpha=0.5, label="hz*", hatch="///", lw=0, where=hz_in_abg < hz_out_abg, zorder=1)

#--- CHZ
ax.plot(phi, chz_in, color="C2", alpha=0.8, lw=2, zorder=3)
ax.plot(phi, chz_out, color="C2", alpha=0.8, lw=2, label="chz", zorder=3)

val1_chz = np.max(chz_in)
val2_chz = np.min(chz_out)
prec1_chz = max(int(-np.log10(val1_chz)) + 2, 0)
prec2_chz = max(int(-np.log10(val2_chz)) + 2, 0)
text_chz = f"CHZ\n{val1_chz:.{prec1_chz}f}$-$\n{val2_chz:.{prec2_chz}f} au"


phi_chz_start = 0 * np.pi / 4
th_chz_start = np.pi / 2
r_chz_start = np.sqrt(np.sum(np.square(chz.zone(theta=th_chz_start, phi=phi_chz_start)[0][i])))
r_chz_end = np.sqrt(np.sum(np.square(chz.zone(theta=th_chz_start, phi=phi_chz_start)[1][i])))

# arrow
if r_chz_end > plot_lim:
    # r_chz_end = r_chz_start + arrow_frac * plot_lim
    r_chz_start = plot_lim * (1 - arrow_frac)
    r_chz_end = plot_lim
    arrow_chz = mpatches.FancyArrowPatch(
        (phi_chz_start, r_chz_start), (phi_chz_start, r_chz_end),
        color="C2", arrowstyle="->", mutation_scale=10, lw=2, zorder=3)
# otherwise just bar
else:
    arrow_chz = mpatches.FancyArrowPatch(
        (phi_chz_start, r_chz_start), (phi_chz_start, r_chz_end),
        color="C2", arrowstyle="-", mutation_scale=10, lw=2, ls=":", zorder=3)
ax.add_patch(arrow_chz)
ax.text(phi_chz_start, r_chz_end, text_chz,
        ha="center", va="bottom", color="C2", fontsize=14, zorder=4)

#--- UHZ
ax.plot(phi, uhz_in, color="C4", alpha=0.8, lw=2, label="uhz", zorder=3)
ax.plot(phi, uhz_out, color="C4", alpha=0.8, lw=2, zorder=3)

val1_uhz = np.max(uhz_in)
val2_uhz = np.min(uhz_out)
prec1_uhz = max(int(-np.log10(val1_uhz)) + 2, 0)
prec2_uhz = max(int(-np.log10(val2_uhz)) + 2, 0)
text_uhz = f"UHZ\n{val1_uhz:.{prec1_uhz}f}$-$\n{val2_uhz:.{prec2_uhz}f} au"

phi_uhz_start = 1 * np.pi / 4
th_uhz_start = np.pi / 2
r_uhz_start = np.sqrt(np.sum(np.square(uhz.zone(theta=th_uhz_start, phi=phi_uhz_start)[0][i])))
r_uhz_end = np.sqrt(np.sum(np.square(uhz.zone(theta=th_uhz_start, phi=phi_uhz_start)[1][i])))

# arrow
if (r_uhz_start >= plot_lim) and (r_uhz_end > plot_lim):
    r_uhz_start = plot_lim * (1 - arrow_frac)
    r_uhz_end = plot_lim
    arrow_uhz = mpatches.FancyArrowPatch(
        (phi_uhz_start, r_uhz_start), (phi_uhz_start, r_uhz_end),
        color="C4", arrowstyle="->", mutation_scale=10, lw=2, zorder=3)
# otherwise just bar
elif (r_uhz_start < plot_lim) and (r_uhz_end > plot_lim):
    r_uhz_end = min(plot_lim, r_uhz_start + plot_lim * arrow_frac)
    arrow_uhz = mpatches.FancyArrowPatch(
        (phi_uhz_start, r_uhz_start), (phi_uhz_start, r_uhz_end),
        color="C4", arrowstyle="->", mutation_scale=10, lw=2, zorder=3)
else:
    arrow_uhz = mpatches.FancyArrowPatch(
        (phi_uhz_start, r_uhz_start), (phi_uhz_start, r_uhz_end),
        color="C4", arrowstyle="-", mutation_scale=10, lw=2, ls=":", zorder=3)
ax.add_patch(arrow_uhz)
ax.text(phi_uhz_start, r_uhz_end, text_uhz,
        ha="center", va="bottom", color="C4", fontsize=14, zorder=4)



#--- ASHZ
ax.plot(phi, ashz_in, color="C1", alpha=0.8, lw=2, label="ashz", zorder=3)

val_ashz = np.max(ashz_in)
prec_ashz = max(int(-np.log10(val_ashz)) + 2, 0)
text_ashz = f"ASHZ\n$>${val_ashz:.{prec_ashz}f} au"

phi_ashz_start = 2 * np.pi / 4
th_ashz_start = np.pi / 2
r_ashz_start = np.sqrt(np.sum(np.square(ashz.zone(theta=th_ashz_start, phi=phi_ashz_start)[0][i])))
r_ashz_end = r_ashz_start + arrow_frac * plot_lim

if r_ashz_start > plot_lim:
    r_ashz_start = plot_lim * (1 - arrow_frac)
    r_ashz_end = plot_lim

arrow_ashz = mpatches.FancyArrowPatch(
    (phi_ashz_start, r_ashz_start), (phi_ashz_start, r_ashz_end),
    color="C1", arrowstyle="->", mutation_scale=10, lw=2, zorder=3)
ax.add_patch(arrow_ashz)
ax.text(phi_ashz_start, r_ashz_end, text_ashz,
        ha="center", va="bottom", color="C1", fontsize=14, zorder=4)

#--- ruler
arrow = mpatches.FancyArrowPatch(path=path_ruler, arrowstyle="|-|", color="black", lw=1, mutation_scale=5, zorder=2)
ax.add_patch(arrow)

val1_ruler = rperi[i]
prec1_ruler = max(int(-np.log10(val1_ruler)) + 2, 0)
if eccen[i] < 1/101:
    ruler_text = f"{val1_ruler:.{prec1_ruler}f} au"
else:
    val2_ruler = rapo[i]
    prec2_ruler = max(int(-np.log10(val2_ruler)) + 2, 0)
    ruler_text = f"{val1_ruler:.{prec1_ruler}f}$-${val2_ruler:.{prec2_ruler}f} au"
ax.text(th_ruler_text, r_ruler_text, ruler_text, ha="center", va="top", color="black", fontsize=14)


# planet marker size scale
def marker_size(mass: float):
    if mass > 1000.0:
        return 1000
    if mass < 0.1:
        return 200
    else:
        return (1000 - 200) * (np.log10(mass) - -1) / (3 - -1) + 200

# iterate through all planets in a system
for r in df_system.iterrows():
    # row as a Series object
    row: pd.Series = r[1]

    a_r = row['pl-orbsmax']
    e_r = row['pl-orbeccen']
    rperi_r = np.abs(a_r * (1 - e_r))
    s_r = np.sign(a_r)
    theta_r = np.pi * (1 + s_r) / 2

    if row['pl-name'] == 'Gor':
        print(a_r, e_r, rperi_r, s_r, theta_r, rperi_r)

    if rperi_r > plot_lim:
        continue

    orbit = ellipse_polar(a_r, e_r, npts)

    # orbit
    ax.plot(orbit[0], orbit[1], lw=1, ls=":", color="black", alpha=0.5, zorder=100)

    if row['plot-label'] == "":
        size_r = marker_size(row['pl-bmasse'])

        # planet marker
        ax.scatter(s_r * theta_r, rperi_r, color="C3", edgecolors="white", lw=0.5, s=size_r, zorder=100)

        # planet text
        ax.text(s_r * theta_r, rperi_r, f"{row['pl-letter']}", ha="center", va="center", color="white", fontsize=14, zorder=101)
        if hostname not in row['pl-name']:
            ax.text(s_r * np.atan2(8 * text_frac, -s_r), (rperi_r**2 + (8 * text_frac * plot_lim)**2)**0.5, f"({row['pl-name']})", ha="center", va="center", color="black", fontsize=14, zorder=101)

    else:
        ax.text(s_r * theta_r, rperi_r, f"{row['plot-label']}", fontsize=16, ha="center", va="center", zorder=100)

# ax.set_yscale("symlog")
ax.set_ylim(0.0, plot_lim)


ax.spines["polar"].set_visible(False)
ax.patch.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# NOTE: set to false if transparent background desired
fig.patch.set_visible(False)
plt.show()

# print planet's HZs
print(f"Orbit: {rperi[i]:.03f}, {rapo[i]:.03f}\n" + \
      f"CHZ: {chz.inner_rad()[i]:.03f}, {chz.outer_rad()[i]:.03f}\n" + \
      f"UHZ: {uhz.inner_rad()[i]:.03f}, {uhz.outer_rad()[i]:.03f}\n" + \
      f"ASHZ: {ashz.inner_rad()[i]:.03f}, {ashz.outer_rad()[i]:.03f}/inf")