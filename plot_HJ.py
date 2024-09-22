from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.colors import to_hex
from matplotlib import colormaps as cm
import numpy as np
import pandas as pd
import re

from matplotlib.colors import Colormap

from alfven_estimates import ra_schrijver, dra_schrijver, RoM, dRoM, RoVK, dRoVK


CMAP1 = ListedColormap([
# CMAP1 = LinearSegmentedColormap.from_list("CMAP1", [
    # "darkred",
    "firebrick",
    # "orangered",
    "orange",
    "yellow",
    "lightyellow",
], "CMAP1")



# CMAP1 = ListedColormap([
# # CMAP1 = LinearSegmentedColormap.from_list("CMAP1", [
#     "darkred",
#     # "firebrick",
#     # "orangered"
#     "orange",
#     "yellow",
#     "lightyellow",
#     # "lightgoldenrodyellow",
#     "lightcyan",
#     "lightskyblue",
#     # "deepskyblue",
#     "dodgerblue"
# ], "CMAP1")

CMAP2 = LinearSegmentedColormap.from_list("CMAP2", [
    # "saddlebrown",
    "darkred",
    "red",
    "orange",
    "yellow"
])

cm.register(cmap=CMAP1)
cm.register(cmap=CMAP2)

COLORS_MPL = ["C0", "C1", "C2", "C3", "C4"]
COLORS_1 = ["red", "blue", "green", "orange", "purple"]
COLORS_2 = ["#DDBB44", "#00AA00", "#DD5500", "#33CCCC", "#000000"]
COLOR_HABITABLE = "black"


CLASS_LABELS = ["subterran", "terran", "superterran", "giant", "no class"]
NAMED_STARS = ["BD-11 4672", "GJ 1132", "GJ 3293", "K2-3", "Kepler-186",
               "Kepler-419", "Kepler-436", "Kepler-438", "Kepler-62", "Kepler-705", "LHS 1140",
               "Proxima Cen", "Ross 128", "Teegarden's Star", "TOI-700",
               "TRAPPIST-1", "Wolf 1061"]
NAMED_PLANETS = ["GJ 3323 c", "K2-3 d", "Kepler-186 f", "Kepler-296 e",
                 "Kepler-438 b", "Kepler-1229 b", "TOI-700 d", "TOI-700 e",
                 "TRAPPIST-1 e", "TRAPPIST-1 f", "TRAPPIST-1 g"] # CHZ terran ASHC>1
SOL_NAMES = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn",
             "Uranus", "Neptune"]


def add_solar_system_planets(df: pd.DataFrame, use_names: bool=False) -> pd.DataFrame:
    ro_sol = 1.85
    dro_sol = 0.26
    ra_sol = 20 * 6.957e8 / 1.496e11
    dra_sol = 0.2 * ra_sol  # 20% error (cycle variation)
    
    peris = [0.307, 0.718, 0.983, 1.381, 4.951, 9.041, 18.29, 29.81]
    masses = [0.0553, 0.815, 1.0, 0.107, 317.8, 95.2, 14.5, 17.1]

    vk_color_sol = 4.81 - 3.27  # Willmer 2018
    
    sol_data = {
        "hostname": ["Sol"] * 8,
        "pl-name": SOL_NAMES,
        "plot-name": SOL_NAMES,
        "Prot": [25] * 8,
        "Ro": [ro_sol] * 8,
        "e_Ro": [dro_sol] * 8,
        "rperi": peris,
        "ASHC": np.divide(peris, ra_sol),
        "e_ASHC": [dra_sol] * 8,
        "st-mass": [1.] * 8,
        "st-lum": [1.] * 8,
        "sy-dist": [1 / 206264.88] * 8,
        "st-age": [4.6] * 8,
        "st-spectype": ["G2 V"] * 8,
        "VK-color": [vk_color_sol] * 8,
        "mass-class": [0, 1, 1, 0, 3, 3, 3, 3],
        "pl-bmasse": masses,
        "pl-bmassj": np.divide(masses, 317.8),
        "habitable": [0, 1, 1, 1, 0, 0, 0, 0]
    }

    if use_names:
        sol_data.update({"plot-name": SOL_NAMES})
        df["plot-name"] = df["plot-name"].apply(lambda x: str(int(x)) if isinstance(x, (float, int)) else x)

    sol = pd.DataFrame(sol_data)

    return pd.concat([df, sol], ignore_index=True)


def _normalized_range(data: np.array, data_range: np.array, shift: float=0., scale: float=1.):
    return shift + scale * (data - np.amin(data_range)) / (np.amax(data_range) - np.amin(data_range))

SPECTYPE_STRS = ["M", "K", "G", "F"]
# SPECTYPE_STRS = ["M", "K", "G", "F", "A", "B", "O"]
SPECTYPE_INDS = {k: v for v, k in enumerate(SPECTYPE_STRS)}
SPECTYPE_VALS = {k: v*10 for v, k in enumerate(SPECTYPE_STRS)}
# SPECTYPE_COLS = {k: v for k, v in zip(["M", "K", "G", "F", "A", "B", "O", ])}
# SPECTZ

def _parse_spectype(spectype: pd.Series):
    p = re.compile(r"^([MKGFABO])(\d(\.\d)?) (I*V?)")

    df = pd.DataFrame(columns=[
        "st-spectype",
        "st-spectype-ltr",
        "st-spectype-num",
        "st-lumiclass",
        "st-spectype-val",
        "st-spectype-qnt",
        "st-spectype-str"
    ])

    df["st-spectype"] = spectype

    for i in df.index:
        st_full = df.loc[i, "st-spectype"]
        
        if pd.isna(st_full):
            continue

        m = re.match(p, st_full)
        if m is not None:
            st = m.group(1)
            num = m.group(2)
            lc = m.group(4)
            ind = SPECTYPE_INDS.get(st, np.nan)
            late_bool = int(float(num)) >= 5 if num is not None else None
            df.loc[i, "st-spectype-ltr"] = st
            df.loc[i, "st-spectype-num"] = float(num)
            df.loc[i, "st-lumiclass"] = lc
            df.loc[i, "st-spectype-val"] = 10 * ind + float(num)
            df.loc[i, "st-spectype-str"] = f"{st}{num} {lc}"
            # df.loc[i, "st-spectype-col"] = np.nan if ind is None else 2 * ind + late_bool
            df.loc[i, "st-spectype-col"] = ind

    # np.quantile()

    # quantiles = np.quantile(df["st-spectype-val"], np.linspace(0.1,1.0,10))


    # df["st-spectype-qnt"] = df["st-spectype-val"] < quantiles

    return df


def plot_fig1(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8,8))
    ax: plt.Axes

    # print(df.count())
    
    
    x = df["pl-orbsmax"]
    x_lo, x_hi = (np.min(x) * 0.9, np.max(x) / 0.9)
    ax.set_xlabel(r"a (AU)", fontsize=18)
    ax.set_xscale("log")
    ax.set_xlim((x_lo, x_hi))

    # y = -2.5 * df["st-lum"] + 5 * np.log10(df["sy-dist"]) + 5
    y = np.power(10, df["st-lum"])/np.square(df["sy-dist"]/4/np.pi)
    # y = -2.5 * df["st-lum"] + 71.197425
    # y *= np.power(206264.8, -2)
    y *= 3.828e26 * np.power(3.0857e16, -2)
    # y *= 1/1368
    y_lo, y_hi = (np.min(y) * 0.9, np.max(y) / 0.9)
    # ax.set_ylabel(r"$\mathcal{F}= \frac{\mathcal{L}}{4 \pi d^2}$ ($\mathcal{L}_\odot$ pc$^{-2}$)", fontsize=18)
    ax.set_ylabel(r"$\mathcal{F}= \frac{\mathcal{L}}{4 \pi d^2}$ ($W\cdot m^{-2}$)", fontsize=18)
    # ax.set_ylabel(r"$\mathcal{F}= \frac{\mathcal{L}}{4 \pi d^2}$ (scaled to the Sun)", fontsize=18)
    ax.set_yscale("log")
    ax.set_ylim((y_lo,y_hi))
    
    z = df["st-age"]
    cmap = cm.get_cmap("cividis_r")

    dots = np.log(df["pl-bmassj"])
    s_shift = 10
    s_scale = 140
    s = _normalized_range(dots, dots, shift=s_shift, scale=s_scale)

    ax.tick_params(labelsize=16, size=5)
    im = ax.scatter(x, y, c=z, marker="o", s=s, cmap=cmap, zorder=2)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"Stellar Age (Gyr)", fontsize=18)
    cb.ax.tick_params(labelsize=16)
    ax.set_title("Hot Jupiters")

    plmvals = np.log([95.2/317.8, 1.])
    plmlabels = [r"Saturn (0.30M$_J$)", r"Jupiter (318M$_J$)"]
    csol = cmap((4.6 - min(z)) / (max(z)-min(z)))
    size_handles = [
        ax.scatter([],[], color=csol, edgecolor="black",
            s=(_normalized_range(plmval, dots, shift=s_shift, scale=s_scale)),
            label=plmlabel)
        for plmlabel, plmval in zip(plmlabels, plmvals)
    ]

    fig.tight_layout()
    ax.legend(handles=size_handles, title=r"Planet Mass ($M_J$)", loc=(0.62,0.01), title_fontsize=13, fontsize=11)
    fig.savefig("imgs/hotjup_dist.png")
    plt.show()


def plot_fig2(df: pd.DataFrame, vals: np.ndarray, use_stmass: bool=False, show_lines: list[int]|None=None):

    # sort by pl_mass so smaller planets don't get covered
    df.sort_values(by="pl-bmasse", ascending=False, inplace=True)
    where_solarsys = df["plot-name"].isin(SOL_NAMES)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    ax: plt.Axes
    cmap = cm.get_cmap("CMAP2" if use_stmass else "plasma_r")

    dots = np.log(df["pl-bmasse"])
    s_shift = 10
    s_scale = 4*35

    x = df["rperi"]
    y = np.abs(df["Prot"])
    z = df["st-mass"] if use_stmass else df["VK_color"]
    s = _normalized_range(dots, dots, shift=s_shift, scale=s_scale)
    
    prot_lo = min(1e0, 0.9 * min(y))
    prot_hi = max(3e2, 1/0.9 * max(y))
    a_lo = min(5e-3, min(x))
    a_hi = None

    ax.set_xlabel(r"a$_p$ (AU)", fontsize=18)
    ax.set_xscale("log")
    ax.set_xlim((a_lo, a_hi))
    ax.set_ylabel(r"P$_{rot}$ (d)", fontsize=18)
    ax.set_yscale("log")
    ax.set_ylim((prot_lo,prot_hi))

    # special plot for solar sys planets
    color_solarsys = cmap((np.array(z[where_solarsys])[0] - min(z)) / (max(z) - min(z)))
    ax.scatter(x[where_solarsys], y[where_solarsys], color=color_solarsys, marker="o", s=s[where_solarsys], zorder=2.5, edgecolor="black")

    im = ax.scatter(x[~where_solarsys], y[~where_solarsys], c=z[~where_solarsys], marker="o", s=s[~where_solarsys], cmap=cmap, zorder=2)
    # im = ax4.scatter(x, y, c=z, marker="o", s=s, cmap=cmap, zorder=2)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"M$_*$ (M$_\odot$)" if use_stmass else r"V-K ($\Delta$mag)", fontsize=18)
    cb.ax.tick_params(labelsize=16)
    
    uncProt = 0.2
    npoints = 500
    protvals = np.linspace(prot_lo, prot_hi, num=npoints)
    nlines = len(vals)
    if use_stmass:
        uncM = 0.05
        prot_grid = pd.Series(np.repeat(protvals, repeats=nlines))
        mass_grid = pd.Series(np.repeat([vals], repeats=npoints, axis=0).flatten())
        dprot_grid = pd.Series([uncProt] * npoints * nlines)
        dmass_grid = pd.Series([uncM] * npoints * nlines)
        rovals = np.reshape(RoM(prot_grid, mass_grid), newshape=(npoints, nlines))
        drovals = np.reshape(dRoM(prot_grid, mass_grid, dprot_grid, dmass_grid), newshape=(npoints, nlines))
        rpvals = np.repeat(protvals, repeats=nlines, axis=0).reshape((npoints,nlines))
        labels = [rf"$M_*=${i:.02f}M$_\odot$" for i in vals]
        c = [cmap((mval - min(z)) / (max(z)-min(z))) for mval in vals]
    else:
        uncVK = 0.1
        prot_grid = pd.Series(np.repeat(protvals, repeats=nlines))
        vk_grid = pd.Series(np.repeat([vals], repeats=npoints, axis=0).flatten())
        dprot_grid = pd.Series([uncProt] * npoints * nlines)
        dvk_grid = pd.Series([uncVK] * npoints * nlines)
        rovals = np.reshape(RoVK(prot_grid, vk_grid), newshape=(npoints, nlines))
        drovals = np.reshape(dRoVK(prot_grid, vk_grid, dprot_grid, dvk_grid), newshape=(npoints, nlines))
        rpvals = np.repeat(protvals, repeats=nlines, axis=0).reshape((npoints,nlines))
        labels = [rf"$V-K=${i:.01f}" for i in vals]
        c = [cmap((vkval - min(z)) / (max(z)-min(z))) for vkval in vals]

    ravals = ra_schrijver(rovals)
    dravals = dra_schrijver(rovals, drovals)
    raupper = ravals + dravals
    ralower = ravals - dravals
    legtitle = "Extent of Mean\n" + r"Alfv$\'e$n Surface"

    ax.set_prop_cycle("color", c)
    if show_lines is None:
        show_lines = range(nlines)
    handles = []
    for i in show_lines:
        hndl, = ax.plot(ravals[:,i], rpvals[:,i], label=labels[i], zorder=1+(i+0.5)/(nlines + 1), color=c[i])
        ax.fill_betweenx(rpvals[:,i], ralower[:,i], raupper[:,i], alpha=0.25, zorder=1+i/(nlines + 1), color=c[i])
        handles.append(hndl)

    # how we break up the pl masses
    plmvals = np.log([95.2, 317.8])
    plmlabels = [r"Saturn (0.30M$_J$)", r"Jupiter (318M$_J$)"]
    csol = cmap((1.0 - min(z)) / (max(z)-min(z)))
    size_handles = [
        ax.scatter([],[], color=csol, edgecolor="black",
            s=(_normalized_range(plmval, dots, shift=s_shift, scale=s_scale)),
            label=plmlabel)
        for plmlabel, plmval in zip(plmlabels, plmvals)
    ]

    ax.tick_params(labelsize=16, size=5)
    
    fig.tight_layout()
    
    legend_lines = ax.legend(handles=handles,
                              title=legtitle,
                              loc=(0.695,0.695),
                              title_fontsize=13,
                              fontsize=11)
    ax.add_artist(legend_lines)
    ax.legend(handles=size_handles, title=r"Planet Mass ($M_J$)", loc=(0.62,0.01), title_fontsize=13, fontsize=11)
    fig.savefig("imgs/hotjup_mass_all.png" if use_stmass else "imgs/hotjup_color_all.png")
    plt.show()


def plot_fig3(df: pd.DataFrame):

    fig, ax = plt.subplots(figsize=(8,8))
    ax: plt.Axes

    x = df["pl-orbsmax"]
    x_lo, x_hi = (np.min(x) * 0.9, np.max(x) / 0.9)
    ax.set_xlabel(r"a (AU)", fontsize=18)
    ax.set_xscale("log")
    ax.set_xlim((x_lo, x_hi))

    # y = -2.5 * df["st-lum"] + 5 * np.log10(df["sy-dist"]) + 5
    y = np.power(10, df["st-lum"])*np.power(df["sy-dist"]*206264, -2)
    # y = -2.5 * df["st-lum"] + 71.197425
    # y *= np.power(206264.8, -2)
    # y *= 3.828e26 * np.power(3.0857e16, -2)
    # y *= 1/1368
    y_lo, y_hi = (np.min(y) * 0.9, np.max(y) / 0.9)
    # ax.set_ylabel(r"$\mathcal{F}= \frac{\mathcal{L}}{4 \pi d^2}$ ($\mathcal{L}_\odot$ pc$^{-2}$)", fontsize=18)
    ax.set_ylabel(r"$\mathcal{F}= \frac{\mathcal{L}}{4 \pi d^2}$ ($W\cdot m^{-2}$)", fontsize=18)
    # ax.set_ylabel(r"$\mathcal{F}= \frac{\mathcal{L}}{4 \pi d^2}$ (scaled to the Sun)", fontsize=18)
    ax.set_yscale("log")
    ax.set_ylim((y_lo,y_hi))
    
    z = df["st-spectype-col"].astype("Int64")
    cmap = cm.get_cmap("CMAP1")

    dots = np.log(df["pl-bmassj"])
    s_shift = 10
    s_scale = 140
    s = _normalized_range(dots, dots, shift=s_shift, scale=s_scale)

    ax.tick_params(labelsize=16, size=5)
    im = ax.scatter(x, y, c=z, cmap=cmap, marker="o",edgecolors="black", s=s, zorder=2, vmin=0, vmax=len(SPECTYPE_STRS)-1)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"Spectral Type", fontsize=18)
    cb_ax = cb.ax
    # cb_ax.tick_params(labelsize=16)
    cb_ax.yaxis.set_ticks([])
    for i, st in enumerate(SPECTYPE_STRS):
        cb_ax.text(0.5, (i + 0.5) * (len(SPECTYPE_STRS) - 1) / len(SPECTYPE_STRS), st, ha="center", va="center")
    cb_ax.yaxis.labelpad = 25
    cb_ax.set_ylabel("Host Star Spectral Type", rotation=270)
    # ax.set_title("Hot Jupiters")

    plmvals = np.log([95.2/317.8, 1.])
    plmlabels = [r"Saturn (0.30M$_J$)", r"Jupiter (318M$_J$)"]
    csol = cmap(2)
    size_handles = [
        ax.scatter([],[], color=csol, edgecolor="black",
            s=(_normalized_range(plmval, dots, shift=s_shift, scale=s_scale)),
            label=plmlabel)
        for plmlabel, plmval in zip(plmlabels, plmvals)
    ]

    fig.tight_layout()
    ax.legend(handles=size_handles, title=r"Planet Mass ($M_J$)", loc=(0.62,0.01), title_fontsize=13, fontsize=11)
    fig.savefig("imgs/hotjup_dist.png")
    plt.show()




if __name__ == "__main__":

    # Load planet habitability and plotting data
    df = pd.read_csv('tables-merged/alfven_data.csv')

    cmap = cm.get_cmap("Paired")
    df_sp = _parse_spectype(df["st-spectype"])
    df = df.merge(df_sp, how="outer", left_index=True, right_index=True, suffixes=(None, "_1"))

    # Load plotting data for HJs
    # df_ss = add_solar_system_planets(df)
    df_ss = df
    criteria = (df_ss["pl-bmasse"] / 317.8 > 0.25) & \
               (df_ss["pl-bmasse"] / 317.8 < 13)
    df_hj = df_ss[criteria].reset_index()

    

    # master plots
    # plot_fig1(df_hj)
    plot_fig3(df_hj)

    # mvals = np.array([0.08 + 1e-5, np.mean(df_hj["st-mass"]), 1.36 - 1e-5])
    # vkvals = np.array([1.1 + 1e-5, np.mean(df_hj["VK-color"]), 7.0 - 1e-5])
    # vals = vkvals
    # plot_fig2(df_hj, vals, use_stmass=False, show_lines=None)
    