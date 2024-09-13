from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps as cm
import numpy as np
import pandas as pd

from alfven_estimates import ra_schrijver, dra_schrijver, RoM, dRoM, RoVK, dRoVK

CMAP2 = LinearSegmentedColormap.from_list("CMAP2", [
    # "saddlebrown",
    "darkred",
    "red",
    "orange",
    "yellow"
])

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
        "pl_name": SOL_NAMES,
        "plot_name": SOL_NAMES,
        "Prot": [25] * 8,
        "Ro": [ro_sol] * 8,
        "e_Ro": [dro_sol] * 8,
        "rperi": peris,
        "ASHC": np.divide(peris, ra_sol),
        "e_ASHC": [dra_sol] * 8,
        "st_mass": [1.] * 8,
        "VK_color": [vk_color_sol] * 8,
        "mass_class": [0, 1, 1, 0, 3, 3, 3, 3],
        "pl_bmasse": masses,
        "pl_bmassj": np.divide(masses, 317.8),
        "habitable": [0, 1, 1, 1, 0, 0, 0, 0]
    }

    if use_names:
        sol_data.update({"plot_name": SOL_NAMES})
        df["plot_name"] = df["plot_name"].apply(lambda x: str(int(x)) if isinstance(x, (float, int)) else x)

    sol = pd.DataFrame(sol_data)

    return pd.concat([df, sol], ignore_index=True)


def _normalized_range(data: np.array, data_range: np.array, shift: float=0., scale: float=1.):
    return shift + scale * (data - np.amin(data_range)) / (np.amax(data_range) - np.amin(data_range))


def plot_fig(df: pd.DataFrame, use_stmass: bool=False):

    # Load plotting data for HJs
    df = add_solar_system_planets(df)
    criteria = (df["pl_bmasse"] / 317.8 > 0.25) & \
               (df["pl_bmasse"] / 317.8 < 13)
    # criteria = (df["habitable"] == 1)
    df = df[criteria].reset_index()
    # df = add_solar_system_planets(df)

    # sort by pl_mass so smaller planets don't get covered
    df.sort_values(by="pl_bmasse", ascending=False, inplace=True)
    where_solarsys = df["plot_name"].isin(SOL_NAMES)
    
    fig, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    ax4: plt.Axes
    cmap = cm.get_cmap("CMAP2" if use_stmass else "plasma_r")

    dots = np.log(df["pl_bmasse"])
    s_shift = 10
    s_scale = 4*35

    x = df["rperi"]
    y = np.abs(df["Prot"])
    z = df["st_mass"] if use_stmass else df["VK_color"]
    s = _normalized_range(dots, dots, shift=s_shift, scale=s_scale)
    
    prot_lo = min(1e0, 0.9 * min(y))
    prot_hi = max(3e2, 1/0.9 * max(y))
    a_lo = min(5e-3, min(x))
    a_hi = None

    ax4.set_xlabel(r"a$_p$ (AU)", fontsize=18)
    ax4.set_xscale("log")
    ax4.set_xlim((a_lo, a_hi))
    ax4.set_ylabel(r"P$_{rot}$ (d)", fontsize=18)
    ax4.set_yscale("log")
    ax4.set_ylim((prot_lo,prot_hi))
    ax4.tick_params(labelsize=16, size=5)

    # special plot for solar sys planets
    color_solarsys = cmap((np.array(z[where_solarsys])[0] - min(z)) / (max(z) - min(z)))
    ax4.scatter(x[where_solarsys], y[where_solarsys], color=color_solarsys, marker="o", s=s[where_solarsys], zorder=2.5, edgecolor="black")

    im = ax4.scatter(x[~where_solarsys], y[~where_solarsys], c=z[~where_solarsys], marker="o", s=s[~where_solarsys], cmap=cmap, zorder=2)
    # im = ax4.scatter(x, y, c=z, marker="o", s=s, cmap=cmap, zorder=2)
    cb = fig.colorbar(im, ax=ax4)
    cb.set_label(r"M$_*$ (M$_\odot$)" if use_stmass else r"V-K ($\Delta$mag)", fontsize=18)
    cb.ax.tick_params(labelsize=16)
    
    uncProt = 0.2
    npoints = 500
    protvals = np.linspace(prot_lo, prot_hi, num=npoints)
    if use_stmass:
        uncM = 0.05
        mvals = np.array([0.08 + 1e-5, np.mean(z), 1.36 - 1e-5])
        nlines = len(mvals)
        prot_grid = pd.Series(np.repeat(protvals, repeats=nlines))
        mass_grid = pd.Series(np.repeat([mvals], repeats=npoints, axis=0).flatten())
        dprot_grid = pd.Series([uncProt] * npoints * nlines)
        dmass_grid = pd.Series([uncM] * npoints * nlines)
        rovals = np.reshape(RoM(prot_grid, mass_grid), newshape=(npoints, nlines))
        drovals = np.reshape(dRoM(prot_grid, mass_grid, dprot_grid, dmass_grid), newshape=(npoints, nlines))
        rpvals = np.repeat(protvals, repeats=nlines, axis=0).reshape((npoints,nlines))
        labels = [rf"$M_*=${i:.02f}M$_\odot$" for i in mvals]
        c = [cmap((mval - min(z)) / (max(z)-min(z))) for mval in mvals]
    else:
        uncVK = 0.1
        vkvals = np.array([1.1 + 1e-5, np.mean(z), 7.0 - 1e-5])
        nlines = len(vkvals)
        prot_grid = pd.Series(np.repeat(protvals, repeats=nlines))
        vk_grid = pd.Series(np.repeat([vkvals], repeats=npoints, axis=0).flatten())
        dprot_grid = pd.Series([uncProt] * npoints * nlines)
        dvk_grid = pd.Series([uncVK] * npoints * nlines)
        rovals = np.reshape(RoVK(prot_grid, vk_grid), newshape=(npoints, nlines))
        drovals = np.reshape(dRoVK(prot_grid, vk_grid, dprot_grid, dvk_grid), newshape=(npoints, nlines))
        rpvals = np.repeat(protvals, repeats=nlines, axis=0).reshape((npoints,nlines))
        labels = [rf"$V-K=${i:.01f}" for i in vkvals]
        c = [cmap((vkval - min(z)) / (max(z)-min(z))) for vkval in vkvals]

    ravals = ra_schrijver(rovals)
    dravals = dra_schrijver(rovals, drovals)
    raupper = ravals + dravals
    ralower = ravals - dravals
    legtitle = "Extent of Mean\n" + r"Alfv$\'e$n Surface"

    ax4.set_prop_cycle("color", c)
    # plot_lines = ax4.plot(ravals, rpvals, label=labels, zorder=1)
    for i in range(nlines):
    # for i in [0]:
        plot_lines = ax4.plot(ravals[:,i], rpvals[:,i], label=labels[i], zorder=1, color=c[i])
        ax4.fill_betweenx(rpvals[:,i], ralower[:,i], raupper[:,i], alpha=0.25, zorder=1.3, color=c[i])

    # how we break up the pl masses
    plmvals = np.log([95.2, 317.8])
    plmlabels = [r"Saturn (0.30M$_J$)", r"Jupiter (318M$_J$)"]
    csol = cmap((1.0 - min(z)) / (max(z)-min(z)))
    size_handles = [
        ax4.scatter([],[], color=csol, edgecolor="black",
            s=(_normalized_range(plmval, dots, shift=s_shift, scale=s_scale)),
            label=plmlabel)
        for plmlabel, plmval in zip(plmlabels, plmvals)
    ]

    ax4.tick_params(labelsize=16, size=5)
    
    fig.tight_layout()
    
    legend_lines = ax4.legend(handles=plot_lines,
                              title=legtitle,
                              loc=(0.695,0.695),
                              title_fontsize=13,
                              fontsize=11)
    ax4.add_artist(legend_lines)
    ax4.legend(handles=size_handles, title=r"Planet Mass ($M_J$)", loc=(0.62,0.01), title_fontsize=13, fontsize=11)
    fig.savefig("imgs/hotjup_mass_all.png" if use_stmass else "imgs/hotjup_color_all.png")
    plt.show()


if __name__ == "__main__":

    # Load planet habitability and plotting data
    df = pd.read_csv('tables-merged/alfven_data.csv')
    df_h = df[df["habitable"] == 1].reset_index()

    # master plots
    plot_fig(df, use_stmass=True)