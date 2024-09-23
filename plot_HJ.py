from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib import colormaps as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import re

from alfven_estimates import ra_schrijver, dra_schrijver, RoM, dRoM, RoVK, dRoVK




CMAP2 = LinearSegmentedColormap.from_list("CMAP2", [
    # "saddlebrown",
    "darkred",
    "red",
    "orange",
    "yellow"
])
CMAP3 = ListedColormap([
    "firebrick",
    "orange",
    "yellow",
    "lightyellow",
], "CMAP3")
CMAP4 = LinearSegmentedColormap.from_list("CMAP4", [
    "firebrick",
    "orange",
    "yellow",
    "lightyellow",
])

cm.register(cmap=CMAP2)
cm.register(cmap=CMAP3)
cm.register(cmap=CMAP4)

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

# SOLAR SYSTEM DATA
SOL_NAMES = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn",
             "Uranus", "Neptune"]
SOL_PERIS = [0.307, 0.718, 0.983, 1.381, 4.951, 9.041, 18.29, 29.81]
SOL_MASSES = [0.0553, 0.815, 1.0, 0.107, 317.8, 95.2, 14.5, 17.1]
SOL_DATA = {
    "hostname": ["Sol"] * 8,
    "pl-name": SOL_NAMES,
    "plot-name": SOL_NAMES,
    "Prot": [25] * 8,
    "rperi": SOL_PERIS,
    "st-mass": [1.] * 8,
    "st-lum": [1.] * 8,
    "st-flux": [1/4 / np.pi] * 8,
    "st-teff": [5772.] * 8,
    "sy-dist": [1 / 206264.88] * 8,
    "st-age": [4.6] * 8,
    "st-spectype": ["G2 V"] * 8,
    "st-spectype-col": [2] * 8,
    "VK-color": [4.81 - 3.27] * 8,   # Willmer 2018
    "pl-bmasse": SOL_MASSES,
    "pl-bmassj": np.divide(SOL_MASSES, 317.8),
    "mass-class": [0, 1, 1, 0, 3, 3, 3, 3],
    "habitable": [0, 1, 1, 1, 0, 0, 0, 0]
}


def add_solar_system_planets(df: pd.DataFrame) -> pd.DataFrame:
    
    sol_data = SOL_DATA.copy()

    ro_sol = 1.85
    dro_sol = 0.26
    ra_sol = 20 * 6.957e8 / 1.496e11
    dra_sol = 0.2 * ra_sol  # 20% error (cycle variation)
    
    sol_data = {**sol_data,
        "Ro": [ro_sol] * 8,
        "e_Ro": [dro_sol] * 8,
        "ASHC": np.divide(SOL_PERIS, ra_sol),
        "e_ASHC": [dra_sol] * 8,
    }

    return pd.concat([df, pd.DataFrame(sol_data)], ignore_index=True)


def _normalized_range(data: np.array, data_range: np.array, shift: float=0., scale: float=1.):
    return shift + scale * (data - np.amin(data_range)) / (np.amax(data_range) - np.amin(data_range))

SPECTYPE_STRS = ["M", "K", "G", "F"]
# SPECTYPE_STRS = ["M", "K", "G", "F", "A", "B", "O"]
SPECTYPE_INDS = {k: v for v, k in enumerate(SPECTYPE_STRS)}
SPECTYPE_VALS = {k: v*10 for v, k in enumerate(SPECTYPE_STRS)}

def _parse_spectype(spectype: pd.Series):
    p = re.compile(r"^([MKGFABO])(\d(\.\d)?)?(\s(V|(IV)|I{1,3}))?")

    df = pd.DataFrame(columns=[
        "st-spectype",
        "st-spectype-ltr",
        "st-spectype-num",
        "st-lumiclass",
        "st-spectype-val",
        "st-spectype-str"
    ])

    df["st-spectype"] = spectype

    for i in df.index:
        st_full = df.loc[i, "st-spectype"]
        
        if pd.isna(st_full):
            continue

        m = re.match(p, st_full)
        if m is not None:
            st_str = ""
            st = m.group(1)
            if st is not None:
                df.loc[i, "st-spectype-ltr"] = st
                st_str += st
            
            num = m.group(2)
            if num is not None:
                ind = SPECTYPE_INDS.get(st, np.nan)
                late_bool = int(float(num)) >= 5 if num is not None else None

                df.loc[i, "st-spectype-num"] = float(num)
                df.loc[i, "st-spectype-val"] = 10 * ind + float(num)
                df.loc[i, "st-spectype-col"] = ind
                st_str += num + " "
            
            lc = m.group(4)
            if lc is not None:
                df.loc[i, "st-lumiclass"] = lc
                st_str += lc

            df.loc[i, "st-spectype-str"] = st_str

    return df


def plot_fig1(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8,8))
    ax: plt.Axes
    
    x = df["pl-orbsmax"]
    x_lo, x_hi = (np.min(x) * 0.9, np.max(x) / 0.9)
    ax.set_xlabel("a (au)", fontsize=18)
    ax.set_xscale("log")
    ax.set_xlim((x_lo, x_hi))

    y = df["sy-dist"]
    y_lo, y_hi = (np.min(y) * 0.9, np.max(y) / 0.9)
    ax.set_ylabel("d (pc)", fontsize=18)
    ax.set_yscale("log")
    ax.set_ylim((y_lo,y_hi))
    
    z = df["st-age"]
    cmap = cm.get_cmap("cividis_r")

    dots = np.log(df["pl-bmassj"])
    s_shift = 10
    s_scale = 140
    s = _normalized_range(dots, dots, shift=s_shift, scale=s_scale)

    ax.tick_params(labelsize=16, size=5)
    im = ax.scatter(x, y, c=z, marker="o", s=s, cmap=cmap, zorder=2, edgecolor="black")
    cb = fig.colorbar(im, ax=ax)
    cb.ax.yaxis.labelpad = 20
    cb.set_label(r"Stellar Age (Gyr)", rotation=270, fontsize=18)
    cb.ax.tick_params(labelsize=16)

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
    

    count = np.sum(1 - np.sum(np.isnan(np.stack((x,y,s,z), axis=1)), axis=1))
    size_handles.insert(0,
        ax.scatter([], [],
                   color="white",
                   edgecolor="black",
                   s=_normalized_range(np.median(dots), dots, shift=s_shift, scale=s_scale),
                   label=rf"Hot Jupiters: {count} ($0.25-13 M_{{J}}$)"))

    ax.legend(handles=size_handles,
                  title=r"Planet Mass ($M_J$)",
                  loc=(0.51,0.845),
                  title_fontsize=13,
                  fontsize=10,
                  labelspacing=0.25)


    fig.savefig("imgs/hotjup_age.png")
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


def plot_teff(df: pd.DataFrame, col: pd.Series|str, title: str, yscale: str, imgname: str):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    ax: plt.Axes


    # ABSCISSA
    x = df["pl-orbsmax"]
    x_lo = 7e-3
    x_hi = 1.3e4
    ax.set_xlabel("a (au)", fontsize=18)
    ax.set_xscale("log")
    ax.set_xlim((x_lo, x_hi))


    # ORDINATE
    if isinstance(col, str):
        col = df[col]
    y = col
    y_lo, y_hi = (np.min(y) * 0.9, np.max(y) / 0.9)
    fig.supylabel(title, fontsize=18)
    ax.set_yscale(yscale)
    ax.set_ylim((y_lo,y_hi))


    # SIZE
    dots = np.log(df["pl-bmassj"])
    s_shift = 10
    s_scale = 140
    s = _normalized_range(dots, dots, shift=s_shift, scale=s_scale)


    # COLOR
    z = df["st-teff"]
    cmap = cm.get_cmap("CMAP4")
    ax.tick_params(labelsize=14, size=5)
    im = ax.scatter(x, y, c=z, cmap=cmap, marker="o", edgecolors="black", s=s, zorder=2)
    cb = fig.colorbar(im, ax=ax)
    cb_ax = cb.ax
    cb_ax.tick_params(labelsize=16)
    cb_ax.yaxis.labelpad = 25
    cb_ax.set_ylabel(r"$T_{eff}$ (K)", rotation=270, fontsize=18)

    plmvals = np.log([95.2/317.8, 1.])
    plmlabels = [r"Saturn (0.30 M$_J$)", r"Jupiter (1 M$_J$)"]
    
    csol = cb.cmap((SOL_DATA["st-teff"][5] - np.min(z)) / (np.max(z) - np.min(z)))
    size_handles = [
        ax.scatter([],[], color=csol, edgecolor="black",
            s=_normalized_range(plmval, dots, shift=s_shift, scale=s_scale),
            label=plmlabel)
        for plmlabel, plmval in zip(plmlabels, plmvals)]

    count = np.sum(1 - np.sum(np.isnan(np.stack((x,y,s,z), axis=1)), axis=1))
    size_handles.insert(0,
        ax.scatter([], [],
                   color="white",
                   edgecolor="black",
                   s=_normalized_range(np.median(dots), dots, shift=s_shift, scale=s_scale),
                   label=rf"Hot Jupiters: {count} ($0.25-13 M_{{J}}$)"))

    ax.legend(handles=size_handles,
                  title=r"Planet Mass ($M_J$)",
                  loc=(0.45,0.845),
                  title_fontsize=13,
                  fontsize=10,
                  labelspacing=0.25)
    
    fig.tight_layout()
    fig.savefig(f"imgs/{imgname}.png")
    plt.show()


def plot_teff_age(df: pd.DataFrame):

    # fig, (ax, ax_lin) = plt.subplots(nrows=2, ncols=1, figsize=(8,8), sharex=True)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    ax: plt.Axes
    divider = make_axes_locatable(ax)
    ax_lin: plt.Axes= divider.append_axes("top", size=5.0, pad=0.0, sharex=ax)

    # ABSCISSA
    x = df["pl-orbsmax"]
    x_lo = 7e-3
    x_hi = 1.3e4
    ax.set_xlabel("a (au)", fontsize=18)
    ax.set_xscale("log")
    ax.set_xlim((x_lo, x_hi))


    # ORDINATE
    y = df["st-age"]
    y_hi = 15
    y_lo = 1e-3
    y_im = 1
    fig.supylabel("Age (Gyr)", fontsize=18)
    ax.set_yscale("log")
    ax.set_ylim((y_lo,y_hi))
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    
    ax.set_yscale('log')
    ax.set_ylim((y_lo, y_im))
    ax_lin.spines["bottom"].set_visible(False)
    ax_lin.xaxis.set_ticks_position("top")
    plt.setp(ax_lin.get_xticklabels(), visible=False)
    ax_lin.set_xscale('linear')
    ax_lin.set_ylim((y_im, y_hi))


    ax.set_xscale("log")
    ax.set_xlim((x_lo, x_hi))
    ax_lin.set_xscale("log")
    ax_lin.set_xlim((x_lo, x_hi))


    # SIZE
    dots = np.log(df["pl-bmassj"])
    s_shift = 10
    s_scale = 140
    s = _normalized_range(dots, dots, shift=s_shift, scale=s_scale)


    # COLOR
    z = df["st-teff"]
    cmap = cm.get_cmap("CMAP4")
    ax.tick_params(labelsize=14, size=5)
    ax_lin.tick_params(labelsize=14, size=5)
    im = ax.scatter(x, y, c=z, cmap=cmap, marker="o",edgecolors="black", s=s, zorder=2)
    ax_lin.scatter(x, y, c=z, cmap=cmap, marker="o",edgecolors="black", s=s, zorder=2)
    cb = fig.colorbar(im, ax=ax)
    cb_ax = cb.ax
    cb_ax.tick_params(labelsize=16)
    cb_ax.yaxis.labelpad = 25
    cb_ax.set_ylabel(r"$T_{eff}$ (K)", rotation=270, fontsize=18)

    plmvals = np.log([95.2/317.8, 1.])
    plmlabels = [r"Saturn (0.30 M$_J$)", r"Jupiter (1 M$_J$)"]
    csol = cb.cmap((SOL_DATA["st-teff"][5] - np.min(z)) / (np.max(z) - np.min(z)))
    size_handles = [
        ax.scatter([],[], color=csol, edgecolor="black",
            s=_normalized_range(plmval, dots, shift=s_shift, scale=s_scale),
            label=plmlabel)
        for plmlabel, plmval in zip(plmlabels, plmvals)
    ]

    count = np.sum(1 - np.sum(np.isnan(np.stack((x,y,s,z), axis=1)), axis=1))
    size_handles.insert(0,
        ax.scatter([], [],
                   color="white",
                   edgecolor="black",
                   s=_normalized_range(np.median(dots), dots, shift=s_shift, scale=s_scale),
                   label=rf"Hot Jupiter: {count} ($0.25-13 M_{{J}}$)"))

    fig.tight_layout()
    ax_lin.legend(handles=size_handles,
                  title=r"Planet Mass ($M_J$)",
                  loc=(0.485,0.78),
                  title_fontsize=13,
                  fontsize=10,
                  labelspacing=0.25)
    fig.savefig("imgs/hotjup_teff_age.png")
    plt.show()


def plot_spec(df: pd.DataFrame, col: pd.Series|str, title: str, yscale: str, imgname: str):

    # fig, (ax, ax_lin) = plt.subplots(nrows=2, ncols=1, figsize=(8,8), sharex=True)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    ax: plt.Axes


    # ABSCISSA
    x = df["pl-orbsmax"]
    x_lo = 7e-3
    x_hi = 1.3e4
    ax.set_xlabel("a (au)", fontsize=18)
    ax.set_xscale("log")
    ax.set_xlim((x_lo, x_hi))


    # ORDINATE
    if isinstance(col, str):
        col = df[col]
    y = col
    y_lo, y_hi = (np.min(y) * 0.9, np.max(y) / 0.9)
    fig.supylabel(title, fontsize=18)
    ax.set_yscale(yscale)
    ax.set_ylim((y_lo,y_hi))


    # SIZE
    dots = np.log(df["pl-bmassj"])
    s_shift = 10
    s_scale = 140
    s = _normalized_range(dots, dots, shift=s_shift, scale=s_scale)


    # COLOR
    z = df["st-spectype-col"].astype("Int64")
    cmap = cm.get_cmap("CMAP3")
    ax.tick_params(labelsize=14, size=5)
    im = ax.scatter(x, y, c=z, cmap=cmap, marker="o",edgecolors="black", s=s, zorder=2)
    cb = fig.colorbar(im, ax=ax)
    cb_ax = cb.ax
    cb_ax.yaxis.set_ticks([])
    for i, st in enumerate(SPECTYPE_STRS):
        cb_ax.text(0.5, (i + 0.5) * (len(SPECTYPE_STRS) - 1) / len(SPECTYPE_STRS), st, ha="center", va="center", fontsize=14)
    cb_ax.yaxis.labelpad = 25
    cb_ax.set_ylabel("Spectral Type", rotation=270, fontsize=18)

    plmvals = np.log([95.2/317.8, 1.])
    plmlabels = [r"Saturn (0.30 M$_J$)", r"Jupiter (1 M$_J$)"]
    csol = cmap(SOL_DATA["st-spectype-col"][5])
    size_handles = [
        ax.scatter([],[], color=csol, edgecolor="black",
            s=_normalized_range(plmval, dots, shift=s_shift, scale=s_scale),
            label=plmlabel)
        for plmlabel, plmval in zip(plmlabels, plmvals)]

    count = np.sum(1 - np.sum(np.isnan(np.stack((x,y,s,z), axis=1)), axis=1))
    size_handles.insert(0,
        ax.scatter([], [],
                   color="white",
                   edgecolor="black",
                   s=_normalized_range(np.median(dots), dots, shift=s_shift, scale=s_scale),
                   label=rf"Hot Jupiters: {count} ($0.25-13 M_{{J}}$)"))

    ax.legend(handles=size_handles,
              title=r"Planet Mass ($M_J$)",
              loc=(0.475,0.845),
              title_fontsize=13,
              fontsize=10,
              labelspacing=0.25)
    
    fig.tight_layout()
    fig.savefig(f"imgs/{imgname}.png")
    plt.show()


def plot_spec_age(df: pd.DataFrame):

    # fig, (ax, ax_lin) = plt.subplots(nrows=2, ncols=1, figsize=(8,8), sharex=True)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    ax: plt.Axes
    divider = make_axes_locatable(ax)
    ax_lin: plt.Axes= divider.append_axes("top", size=5.0, pad=0.0, sharex=ax)

    # ABSCISSA
    x = df["pl-orbsmax"]
    x_lo = 7e-3
    x_hi = 1.3e4
    ax.set_xlabel("a (au)", fontsize=18)
    ax.set_xscale("log")


    # ORDINATE
    y = df["st-age"]
    y_hi = 15
    y_lo = 1e-3
    y_im = 1
    fig.supylabel("Age (Gyr)", fontsize=18)
    ax.set_yscale("log")
    ax.set_ylim((y_lo,y_hi))
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    
    ax.set_yscale('log')
    ax.set_ylim((y_lo, y_im))
    ax_lin.spines["bottom"].set_visible(False)
    ax_lin.xaxis.set_ticks_position("top")
    plt.setp(ax_lin.get_xticklabels(), visible=False)
    ax_lin.set_xscale('linear')
    ax_lin.set_ylim((y_im, y_hi))


    ax.set_xscale("log")
    ax.set_xlim((x_lo, x_hi))
    ax_lin.set_xscale("log")
    ax_lin.set_xlim((x_lo, x_hi))


    # SIZE
    dots = np.log(df["pl-bmassj"])
    s_shift = 10
    s_scale = 140
    s = _normalized_range(dots, dots, shift=s_shift, scale=s_scale)


    # COLOR
    z = df["st-spectype-col"].astype("Int64")
    cmap = cm.get_cmap("CMAP3")
    ax.tick_params(labelsize=14, size=5)
    ax_lin.tick_params(labelsize=14, size=5)
    im = ax.scatter(x, y, c=z, cmap=cmap, marker="o",edgecolors="black", s=s, zorder=2, vmin=0, vmax=len(SPECTYPE_STRS)-1)
    ax_lin.scatter(x, y, c=z, cmap=cmap, marker="o",edgecolors="black", s=s, zorder=2, vmin=0, vmax=len(SPECTYPE_STRS)-1)
    cb = fig.colorbar(im, ax=ax)
    cb_ax = cb.ax
    cb_ax.yaxis.set_ticks([])
    for i, st in enumerate(SPECTYPE_STRS):
        cb_ax.text(0.5, (i + 0.5) * (len(SPECTYPE_STRS) - 1) / len(SPECTYPE_STRS), st, ha="center", va="center", fontsize=14)
    cb_ax.yaxis.labelpad = 25
    cb_ax.set_ylabel("Spectral Type", rotation=270, fontsize=18)

    plmvals = np.log([95.2/317.8, 1.])
    plmlabels = [r"Saturn (0.30 M$_J$)", r"Jupiter (1 M$_J$)"]
    csol = cmap(SOL_DATA["st-spectype-col"][5])
    size_handles = [
        ax.scatter([],[], color=csol, edgecolor="black",
            s=_normalized_range(plmval, dots, shift=s_shift, scale=s_scale),
            label=plmlabel)
        for plmlabel, plmval in zip(plmlabels, plmvals)]
    
    count = np.sum(1 - np.sum(np.isnan(np.stack((x,y,s,z), axis=1)), axis=1))
    size_handles.insert(0,
        ax.scatter([], [],
                   color="white",
                   edgecolor="black",
                   s=_normalized_range(np.median(dots), dots, shift=s_shift, scale=s_scale),
                   label=rf"Hot Jupiter: {count} ($0.25-13 M_{{J}}$)"))

    ax_lin.legend(handles=size_handles,
                  title=r"Planet Mass ($M_J$)",
                  loc=(0.51,0.78),
                  title_fontsize=13,
                  fontsize=10,
                  labelspacing=0.25)
    
    fig.tight_layout()
    fig.savefig("imgs/hotjup_spectype_age.png")
    plt.show()


if __name__ == "__main__":

    add_solar: bool=False

    # Load planet habitability and plotting data
    df = pd.read_csv('tables-merged/alfven_data.csv')
    
    if add_solar:
        df = add_solar_system_planets(df)

    df["st-flux"] = df["st-lum"] - 2 * np.log10(df["sy-dist"]*206264) - np.log10(4*np.pi)

    df_sp = _parse_spectype(df["st-spectype"])
    df = df.merge(df_sp, how="outer", left_index=True, right_index=True, suffixes=(None, "_1"))

    # Load plotting data for HJs    
    criteria = (df["pl-bmasse"] / 317.8 > 0.25) & \
               (df["pl-bmasse"] / 317.8 < 13)
    df_hj = df[criteria].reset_index()

    # master plots
    plot_fig1(df_hj)
    
    # title = r"d (pc)"
    # plot_teff(df_hj, "sy-dist", title, "log", "hotjup_teff_dist")
    # title = r"$\mathcal{L}$ ($\mathcal{L}_\odot$)"
    # plot_teff(df_hj, np.power(10, df_hj["st-lum"]), title, "log", "hotjup_teff_lum")
    # title = r"$\mathcal{F}=\frac{\mathcal{L}}{4 \pi d^2}$ ($\mathcal{L}_\odot$ au$^{-2}$)"
    # plot_teff(df_hj, np.power(10, df_hj["st-flux"]), title, "log", "hotjup_teff_flux")
    # plot_teff_age(df_hj)


    # title = r"d (pc)"
    # plot_spec(df_hj, "sy-dist", title, "log", "hotjup_spectype_dist")
    # title = r"$\mathcal{L}$ ($\mathcal{L}_\odot$)"
    # plot_spec(df_hj, np.power(10, df_hj["st-lum"]), title, "log", "hotjup_spectype_lum")
    # title = r"$\mathcal{F}=\frac{\mathcal{L}}{4 \pi d^2}$ ($\mathcal{L}_\odot$ au$^{-2}$)"
    # plot_spec(df_hj, np.power(10, df_hj["st-flux"]), title, "log", "hotjup_spectype_flux")
    # plot_spec_age(df_hj)
    

    # mvals = np.array([0.08 + 1e-5, np.mean(df_hj["st-mass"]), 1.36 - 1e-5])
    # vkvals = np.array([1.1 + 1e-5, np.mean(df_hj["VK-color"]), 7.0 - 1e-5])
    # vals = vkvals
    # plot_fig2(df_hj, vals, use_stmass=False, show_lines=None)
    