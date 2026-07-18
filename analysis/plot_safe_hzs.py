import argparse as ap
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from hz import CircumstellarHabitableZone, AlfvenSurfaceHabitableZone
from hz.uhz import UltravioletHabitableZoneABG

F_NUV = 0.1
LUM_CLASS_FILL = 5



parser = ap.ArgumentParser(prog="plot_safe_hzs.py",
                           usage="%(prog)s [options]",
                           description="",
                           formatter_class=ap.ArgumentDefaultsHelpFormatter)

# parser.add_argument("plname")
parser.add_argument("--fnuv", default=F_NUV, type=float,
                    help="transmission of NUV emission through planetary atmosphere.")

parser.add_argument("-c", "--conditions", default="df['pl_name'] != ''",
                    help="conditions filtering planets. Use 'df' to reference the dataframe; numpy functions are available.")
parser.add_argument("-l", "--lum_class_fill", default=LUM_CLASS_FILL, type=float,
                    help="fill value for luminosity class when missing (typically 5=V/Main Sequence).")



args = vars(parser.parse_args())

cond_s = args["conditions"]
fnuv_val = args["fnuv"]
lum_class_fill = args["lum_class_fill"]

df = pd.read_csv("../db/params-rotp.csv")


# determine filtering conditions
cond = eval(cond_s)

r_lim = 100.
df = df.loc[cond].reset_index()

lumi = df["st_lum"].to_numpy()
teff = df["st_teff"].to_numpy()
prot = df["st_rotp"].to_numpy()
plmass = df["pl_bmasse"].to_numpy()
rperi = (1 - df["pl_orbeccen"].to_numpy()) * df["pl_orbsmax"].to_numpy()
rapo = (1 + df["pl_orbeccen"].to_numpy()) * df["pl_orbsmax"].to_numpy()
eccen = df["pl_orbeccen"].to_numpy()
smax = df["pl_orbsmax"].to_numpy()

fnuv = np.repeat([fnuv_val], repeats=len(df))
lumclass = df["st_lc_num"].fillna(lum_class_fill).to_numpy()

def ellipse(a, e, theta):
    r = a * (1 - e**2) / (1 - e * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=0)

def ellipse_polar(a, e, npts: int):
    theta = np.linspace(0, 2 * np.pi, npts, endpoint=True)

    ones_theta = np.ones_like(theta)
    ones_df = np.ones_like(a)

    a_rep = np.outer(a, ones_theta)
    e_rep = np.outer(e, ones_theta)
    theta_rep = np.outer(ones_df, theta)
    # if nrows is not None:
    #     theta = np.repeat([theta], repeats=nrows, axis=0).T
    r_rep = a_rep * (1 - e_rep**2) / (1 - e_rep * np.cos(theta_rep))
    return np.stack([theta_rep, r_rep], axis=0)


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

# chz_lo, chz_hi = chz.zone(theta, phi)
# uhz_lo, uhz_hi = uhz.zone(theta, phi)
# ashz_lo, ashz_hi = ashz.zone(theta, phi)

chz_inner = chz.inner_rad()
chz_outer = chz.outer_rad()
uhz_inner = uhz.inner_rad()
uhz_outer = uhz.outer_rad()
ashz_inner = ashz.inner_rad()
ashz_outer = ashz.outer_rad()
hz_inner_abg = np.max([chz_inner, uhz_inner, ashz_inner], axis=0)
hz_outer_abg = np.min([chz_outer, uhz_outer, ashz_outer], axis=0)



def calc_time_safe(r, inner, outer):
    inner_rep = np.repeat([inner], r.shape[-1], axis=0).T
    outer_rep = np.repeat([outer], r.shape[-1], axis=0).T
    time_safe = np.sum((r > inner_rep) & (r < outer_rep), axis=1) / npts
    return time_safe

orbits = ellipse_polar(smax, eccen, npts)
orbits_r = orbits[1]

chz_safe = calc_time_safe(orbits_r, chz_inner, chz_outer)
uhz_safe = calc_time_safe(orbits_r, uhz_inner, uhz_outer)
ashz_safe = calc_time_safe(orbits_r, ashz_inner, ashz_outer)
hz_safe = calc_time_safe(orbits_r, hz_inner_abg, hz_outer_abg)

safe_cand = hz_safe & (df["disposition"] == 0)
safe_conf = hz_safe & (df["disposition"] == 1)

print(f"Safe candidate planets: {np.sum(safe_cand)}")
print(f"Safe confirmed planets: {np.sum(safe_conf)}")

df_chz_safe = df[chz_safe > 0.0]
df_uhz_safe = df[uhz_safe > 0.0]
df_ashz_safe = df[ashz_safe > 0.0]
df_safe = df[hz_safe > 0.0]

df_sol = pd.read_csv("../db/solar_system.csv")
earth_row = df_sol.loc[df_sol["pl_name"] == "Earth"].iloc[0]

def safe_plot(df: pd.DataFrame, col1: str, col2: str, title: str=None,
              col1_label: str=None, col2_label: str=None,
              chz_cond=None, uhz_cond=None, ashz_cond=None, hz_cond=None,
              filename: str=None):

    if col1_label is None:
        col1_label = col1

    if col2_label is None:
        col2_label = col2

    if chz_cond is None:
        chz_cond = chz_safe > 0.0

    if uhz_cond is None:
        uhz_cond = uhz_safe > 0.0

    if ashz_cond is None:
        ashz_cond = ashz_safe > 0.0

    if hz_cond is None:
        hz_cond = hz_safe > 0.0

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(8,8), sharex=True, sharey=True)

    ax1.set_title(f"CHZ ({np.sum(chz_cond)})", fontsize=16)
    ax1.scatter(df.loc[chz_cond, col1], df.loc[chz_cond, col2], c="C2", s=10)
    ax1.text(earth_row[col1], earth_row[col2], earth_row["plot_label"], fontsize=24, color="C2", va="center", ha="center")

    ax2.set_title(f"UHZ ({np.sum(uhz_cond)})", fontsize=16)
    ax2.scatter(df.loc[uhz_cond, col1], df.loc[uhz_cond, col2], c="C4", s=10)
    ax2.text(earth_row[col1], earth_row[col2], earth_row["plot_label"], fontsize=24, color="C4", va="center", ha="center")

    ax3.set_title(f"ASHZ ({np.sum(ashz_cond)})", fontsize=16)
    ax3.scatter(df.loc[ashz_cond, col1], df.loc[ashz_cond, col2], c="C1", s=10)
    ax3.text(earth_row[col1], earth_row[col2], earth_row["plot_label"], fontsize=24, color="C1", va="center", ha="center")

    ax4.set_title(f"HZ ({np.sum(hz_cond)})", fontsize=16)
    ax4.scatter(df.loc[hz_cond, col1], df.loc[hz_cond, col2], c="C0", s=10)
    ax4.text(earth_row[col1], earth_row[col2], earth_row["plot_label"], fontsize=24, color="C0", va="center", ha="center")

    ax1.tick_params(labelsize=14)
    ax2.tick_params(labelsize=14)
    ax3.tick_params(labelsize=14)
    ax4.tick_params(labelsize=14)
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    if title is not None:
        fig.suptitle(title, fontsize=20)

    fig.supxlabel(col1_label, fontsize=16)
    fig.supylabel(col2_label, fontsize=16)

    fig.tight_layout()

    # NOTE: set to `False` if transparent background desired
    ax1.patch.set_visible(False)
    ax2.patch.set_visible(False)
    ax3.patch.set_visible(False)
    ax4.patch.set_visible(False)
    fig.patch.set_visible(False)

    if filename is not None:
        fig.savefig(filename)

    plt.show()

safe_plot(df, "pl_orbsmax", "pl_bmasse",
          title="Marginally-Safe Planets and Candidates",
          col1_label="Semi-major Axis (au)",
          col2_label=r"Planet Mass (M$_\oplus$)",
          filename="./plots/marginally-safe_hzs.png")

safe_plot(df, "pl_orbsmax", "pl_bmasse",
          title="Safe Planets and Candidates",
          col1_label="Semi-major Axis (au)",
          col2_label=r"Planet Mass (M$_\oplus$)",
          chz_cond=chz_safe == 1.0,
          uhz_cond=uhz_safe == 1.0,
          ashz_cond=ashz_safe == 1.0,
          hz_cond=hz_safe == 1.0,
          filename="./plots/safe_hzs.png")
