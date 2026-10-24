import argparse as ap
import numpy as np
import pandas as pd

from hz import CircumstellarHabitableZone
from hz import UltravioletHabitableZoneABG
from hz import AlfvenSurfaceHabitableZone

F_NUV = 0.1
LUM_CLASS_FILL = 5


# command-line argument parsing
parser = ap.ArgumentParser(prog="stats_hz.py",
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


# fetch command-line arguments
args = vars(parser.parse_args())

cond_s = args["conditions"]
fnuv_val = args["fnuv"]
lum_class_fill = args["lum_class_fill"]

# load system parameters
df = pd.read_csv("../db/params-step3.csv")


# determine filtering conditions from command-line input
cond = eval(cond_s)
df = df.loc[cond].reset_index()

# radial extent for ASHZ (required for ASHZ)
r_lim = 100.

# define simple aliases for parameters necessary to calculate HZs
lumi = df["st_lum"].to_numpy()
teff = df["st_teff"].to_numpy()
prot = df["st_rotp"].to_numpy()
plmass = df["pl_bmasse"].to_numpy()
rperi = (1 - df["pl_orbeccen"].to_numpy()) * df["pl_orbsmax"].to_numpy()
rapo = (1 + df["pl_orbeccen"].to_numpy()) * df["pl_orbsmax"].to_numpy()
eccen = df["pl_orbeccen"].to_numpy()
smax = df["pl_orbsmax"].to_numpy()

# define a priori the planetary NUV transmission factor
fnuv = np.repeat([fnuv_val], repeats=len(df))

# define a priori the stellar LC for stars lacking this datum
lumclass = df["st_lc_num"].fillna(lum_class_fill).to_numpy()



# NOT USED
def ellipse(a, e, theta):
    """
    Calculate planet orbits in 2D Cartesian (x, y) coordinates given semi-major
    axis and eccentricity data. Samples the planet orbit according to the
    provided range of angles `theta`.
    """
    r = a * (1 - e**2) / (1 - e * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=0)

def ellipse_polar(a, e, npts: int):
    """
    Calculate planet orbits in 2D polar (r, theta) coordinates given semi-major
    axis and eccentricity data. Samples the planet orbit evenly in theta with
    `npts` number of points.
    """
    theta = np.linspace(0, 2 * np.pi, npts, endpoint=True)

    ones_theta = np.ones_like(theta)
    ones_df = np.ones_like(a)

    a_rep = np.outer(a, ones_theta)
    e_rep = np.outer(e, ones_theta)
    theta_rep = np.outer(ones_df, theta)

    r_rep = a_rep * (1 - e_rep**2) / (1 - e_rep * np.cos(theta_rep))
    return np.stack([theta_rep, r_rep], axis=0)



### CREATE HZ CALCULATORS

# Kopparapu+ 14 CHZ
chz = CircumstellarHabitableZone(teff, 10**lumi, plmass)

# Spinelli+ 23 UHZ
uhz = UltravioletHabitableZoneABG(teff, fnuv)

# Atkinson+ 24 ASHZ
# r_lim = np.max(rapo)
# print(f"Max radius: {r_lim:.03f} au")
ashz = AlfvenSurfaceHabitableZone(teff, lumclass, prot, r_lim)



### CALCULATE INNER AND OUTER RADII FOR EACH HZ

chz_inner = chz.inner_rad()
chz_outer = chz.outer_rad()
uhz_inner = uhz.inner_rad()
uhz_outer = uhz.outer_rad()
ashz_inner = ashz.inner_rad()
ashz_outer = ashz.outer_rad()
hz_inner_abg = np.max([chz_inner, uhz_inner, ashz_inner], axis=0)
hz_outer_abg = np.min([chz_outer, uhz_outer, ashz_outer], axis=0)



# NOT USED
def calc_angles_safe(r: np.ndarray,
                     theta: np.ndarray,
                     orb_a: np.ndarray,
                     orb_e: np.ndarray,
                     inner: np.ndarray,
                     outer: np.ndarray):
    """
    Calculate the fraction of a planet's orbital angle (2 pi) that a planet
    spends within a HZ.
    """
    inner_rep = np.repeat([inner], r.shape[-1], axis=0).T
    outer_rep = np.repeat([outer], r.shape[-1], axis=0).T
    time_safe = np.sum((r > inner_rep) & (r < outer_rep), axis=1) / npts
    return time_safe

def calc_time_safe(r: np.ndarray,
                   theta: np.ndarray,
                   orb_a: np.ndarray,
                   orb_e: np.ndarray,
                   inner: np.ndarray,
                   outer: np.ndarray,
                   atol: np.float64=1e-6):
    """
    Calculate the fraction of a planet's orbital period that a planet spends
    within a HZ.
    """
    inner_rep = np.repeat([inner], r.shape[-1], axis=0).T
    outer_rep = np.repeat([outer], r.shape[-1], axis=0).T

    ind_unsafe = (r <= inner_rep) | (r >= outer_rep)
    ind_nan = np.isnan(inner_rep) | np.isnan(outer_rep)

    r_copy = r.copy()
    r_copy[ind_unsafe] = 0.0
    r_copy[ind_nan] = np.nan
    r_copy = r_copy[:,:-1]

    dtheta = np.diff(theta)

    # time in zone (tiz)
    factor = 2*np.pi * (orb_a ** 2) * np.sqrt(1 - orb_e ** 2)
    sum_rad = np.sum(r_copy**2 * dtheta, axis=1)
    tiz = sum_rad / factor
    tiz[np.isclose(tiz, 1.0, atol=atol)] = 1.0
    return tiz

# number of points to sample planet orbit
npts = 1000
# calculate (polar) orbits of each planet around star
orbits = ellipse_polar(smax, eccen, npts)
# orbit angular coordinate
orbits_th = orbits[0]
# orbit radial coordinate
orbits_r = orbits[1]

# determine the amount of time (not angle) of a planet's orbit lies in HZ
chz_safe = calc_time_safe(orbits_r, orbits_th, smax, eccen, chz_inner, chz_outer)
uhz_safe = calc_time_safe(orbits_r, orbits_th, smax, eccen, uhz_inner, uhz_outer)
ashz_safe = calc_time_safe(orbits_r, orbits_th, smax, eccen, ashz_inner, ashz_outer)
hz_safe = calc_time_safe(orbits_r, orbits_th, smax, eccen, hz_inner_abg, hz_outer_abg)

# indexes CANDIDATE planets that spend non-zero time in mutual HZ
safe_cand = hz_safe & (df["disposition"] == 0)
# indexes CONFIRMED planets that spend non-zero time in mutual HZ
safe_conf = hz_safe & (df["disposition"] == 1)

# report statistics of 'safe' planets
print(f"Safe candidate planets: {np.sum(safe_cand)}")
print(f"Safe confirmed planets: {np.sum(safe_conf)}")

print("chz > 0", np.sum(chz_safe > 0.0))
print("uhz > 0", np.sum(uhz_safe > 0.0))
print("ashz > 0", np.sum(ashz_safe > 0.0))
print("hz > 0", np.sum(hz_safe > 0.0))

print("chz = 1", np.sum(chz_safe == 1.0))
print("uhz = 1", np.sum(uhz_safe == 1.0))
print("ashz = 1", np.sum(ashz_safe == 1.0))
print("hz = 1", np.sum(hz_safe == 1.0))

df_chz_safe = df[chz_safe > 0.0]
df_uhz_safe = df[uhz_safe > 0.0]
df_ashz_safe = df[ashz_safe > 0.0]
df_safe = df[hz_safe > 0.0]

# print names of planets that spend non-zero time in HZ
for i, pl in df_safe["pl_name"].items():
    print(f"{pl}\t{hz_safe[i]:.3f}")