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


def make_plot_names_pl(df: pd.DataFrame,
                       named_planets: list[str]=NAMED_PLANETS) -> pd.DataFrame:
    
    pl_names = pd.Series(named_planets, name="pl_name")

    df_fltr = pd.merge(df, pl_names, how="inner", on="pl_name").sort_values(by="ASHC")[["pl_name"]]
    df_fltr["plot_name"] = np.asarray(df_fltr.groupby("pl_name", sort=False).ngroup() + 1, dtype=str)
    df_map = df_fltr[["plot_name", "pl_name"]]

    df_all_plot_name = pd.merge(df, df_map, how="left", on="pl_name")["plot_name"]
    df_all_plot_name.fillna("", inplace=True)

    return df_all_plot_name


def add_solar_system_planets(df: pd.DataFrame, use_names: bool=False) -> pd.DataFrame:
    ro_sol = 1.85
    dro_sol = 0.26
    ra_sol = 20 * 6.957e8 / 1.496e11
    dra_sol = 0.2 * ra_sol  # 20% error (cycle variation)
    r_p_venus = 0.718
    r_p_earth = 0.983
    r_p_mars = 1.381
    mass_venus = 0.815
    mass_earth = 1.0
    mass_mars = 0.107
    vk_color_sol = 4.81 - 3.27  # Willmer 2018
    
    sol_data = {
        "hostname": ["Sol", "Sol", "Sol"],
        "pl_name": ["Venus", "Earth", "Mars"],
        "plot_name": ["Venus", "Earth", "Mars"],
        "Prot": [25, 25, 25],
        "Ro": [ro_sol, ro_sol, ro_sol],
        "e_Ro": [dro_sol, dro_sol, dro_sol],
        "rperi": [r_p_venus, r_p_earth, r_p_mars],
        "ASHC": [r_p_venus / ra_sol, r_p_earth / ra_sol, r_p_mars / ra_sol],
        "e_ASHC": [dra_sol, dra_sol, dra_sol],
        "st_mass": [1., 1., 1.],
        "VK_color": [vk_color_sol, vk_color_sol, vk_color_sol],
        "mass_class": [1, 1, 0],
        "pl_bmasse": [mass_venus, mass_earth, mass_mars],
        "habitable": [1, 1, 1]
    }

    if use_names:
        sol_data.update({"plot_name": ["Venus", "Earth", "Mars"]})
        df["plot_name"] = df["plot_name"].apply(lambda x: str(int(x)) if isinstance(x, (float, int)) else x)

    sol = pd.DataFrame(sol_data)

    return pd.concat([df, sol], ignore_index=True)


def _normalized_range(data: np.array, data_range: np.array, shift: float=0., scale: float=1.):
    return shift + scale * (data - np.amin(data_range)) / (np.amax(data_range) - np.amin(data_range))


def _named_text_posn_fig4(df: pd.DataFrame):
    
    df["xtext"] = 10
    df["ytext"] = 10

    df.loc[df["plot_name"] == "1", "xtext"] = 15
    df.loc[df["plot_name"] == "1", "ytext"] = 5

    df.loc[df["plot_name"] == "2", "xtext"] = 9
    df.loc[df["plot_name"] == "2", "ytext"] = -11

    df.loc[df["plot_name"] == "3", "xtext"] = 17
    df.loc[df["plot_name"] == "3", "ytext"] = 3

    df.loc[df["plot_name"] == "5", "xtext"] = 7
    df.loc[df["plot_name"] == "5", "ytext"] = -13

    df.loc[df["plot_name"] == "6", "xtext"] = 19
    df.loc[df["plot_name"] == "6", "ytext"] = 1

    df.loc[df["plot_name"] == "8", "xtext"] = 15
    df.loc[df["plot_name"] == "8", "ytext"] = 5

    df.loc[df["plot_name"] == "10", "xtext"] = 5
    df.loc[df["plot_name"] == "10", "ytext"] = 15

    df.loc[df["plot_name"] == "11", "xtext"] = 7
    df.loc[df["plot_name"] == "11", "ytext"] = -13
    
    df.loc[df["plot_name"] == "Venus", "xtext"] = 5
    df.loc[df["plot_name"] == "Venus", "ytext"] = -15
    
    df.loc[df["plot_name"] == "Mars", "xtext"] = 13
    df.loc[df["plot_name"] == "Mars", "ytext"] = -7

    df.loc[df["plot_name"] == "Earth", "xtext"] = 15
    df.loc[df["plot_name"] == "Earth", "ytext"] = 5

    return df


def plot_fig1(df: pd.DataFrame, classcol: str="mass_class"):
    
    df = df.copy(deep=False)
    
    x1lo = 0.0
    x1hi = 1.75
    x2lo = 0.5
    x2hi = 8.0
    ylo = 7.5e-2
    yhi = 100

    # exclude outlier data in ASHC and VK_color
    df.loc[~(
        (df["ASHC"] >= ylo) &
        (df["ASHC"] <= yhi) &
        (df["VK_color"] >= x2lo) &
        (df["VK_color"] <= x2hi)),
        "VK_color"] = np.nan
    
    # exclude outlier data in ASHC and st_mass
    df.loc[~(
        (df["ASHC"] >= ylo) &
        (df["ASHC"] <= yhi) &
        (df["st_mass"] >= x1lo) &
        (df["st_mass"] <= x1hi)),
        "st_mass"] = np.nan

    colors = ["red", "green", "mediumorchid", "orange", "black"]
    grp_counts_sm = df.groupby(by=classcol).count()["st_mass"]
    grp_counts_vk = df.groupby(by=classcol).count()["VK_color"]
    grp_names = ["subterran", "terran", "superterran", "giant", "no class"]

    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
    ax1: plt.Axes
    ax2: plt.Axes
    
    df_nh = df[df["habitable"] == 0].reset_index(inplace=False)
    df_u, df_inv = np.unique(df_nh[classcol], return_inverse=True, axis=0)

    alpha_nh = 0.8
    # Plot by planet class
    for idx in reversed(range(max(df_inv) + 1)):
        grp = df_u[idx]
        temp = df_nh[df_inv == idx]
        label_sm = f"{grp_names[grp]} ({grp_counts_sm[grp]})"
        label_vk = f"{grp_names[grp]} ({grp_counts_vk[grp]})"
        
        ax1.plot(np.array(temp["st_mass"]), np.array(temp["ASHC"]),
                 c=colors[grp], linestyle="none", marker="+", fillstyle="none", alpha=alpha_nh, label=label_sm)
        ax2.plot(np.array(temp["VK_color"]), np.array(temp["ASHC"]),
                 c=colors[grp], linestyle="none", marker="+", fillstyle="none", alpha=alpha_nh, label=label_vk)

    df_h = df[df["habitable"] == 1].reset_index(inplace=False)
    df_u, df_inv = np.unique(df_h[classcol], return_inverse=True, axis=0)

    for idx in reversed(range(max(df_inv) + 1)):
        grp = df_u[idx]
        temp = df_h[df_inv == idx]

        # maybes_where = ((temp["ASHC"] - temp["e_ASHC"] < 1) & (temp["ASHC"] > 1)) | ((temp["ASHC"] + temp["e_ASHC"] > 1) & (temp["ASHC"] < 1))
        maybes_where = (temp["pl_name"] == "TRAPPIST-1 e") | (temp["pl_name"] == "GJ 1002 c") | (temp["pl_name"] == "GJ 273 b")
        maybes = temp[maybes_where]
        certain = temp[~maybes_where]

        ax1.plot(np.array(certain["st_mass"]), np.array(certain["ASHC"]),
                 c=colors[grp], linestyle="none", marker="o", fillstyle="full")
        ax2.plot(np.array(certain["VK_color"]), np.array(certain["ASHC"]),
                 c=colors[grp], linestyle="none", marker="o", fillstyle="full")

        ax1.plot(np.array(maybes["st_mass"]), np.array(maybes["ASHC"]),
                 c=colors[grp], linestyle="none", marker="o", fillstyle="none", markeredgecolor=colors[grp])
        ax2.plot(np.array(maybes["VK_color"]), np.array(maybes["ASHC"]),
                 c=colors[grp], linestyle="none", marker="o", fillstyle="none", markeredgecolor=colors[grp])

    # Legend markers for OHZ
    ax1.plot([], [], linestyle="none", marker="o", fillstyle="full", alpha=alpha_nh, color="green", markeredgecolor="green", label=f"OHZ ({len(df_h)})")
    ax2.plot([], [], linestyle="none", marker="o", fillstyle="full", alpha=alpha_nh, color="green", markeredgecolor="green", label=f"OHZ ({len(df_h)})")

    # Legend markers for 'maybes'
    ax1.plot([], [], linestyle="none", marker="o", fillstyle="none", alpha=alpha_nh, markeredgecolor="green", label=r"ASHC $\gtrsim$ 1")
    ax2.plot([], [], linestyle="none", marker="o", fillstyle="none", alpha=alpha_nh, markeredgecolor="green", label=r"ASHC $\gtrsim$ 1")
    
    
    ax1.hlines(1.0, xmin=x1lo, xmax=x1hi, linestyles="dashed", linewidth=1.0,
              colors="gray")
    ax1.set_ylabel("ASHC", fontsize=18)
    ax1.set_yscale("log")
    ax1.set_ylim((ylo, yhi))
    ax1.set_xlabel(r"Stellar Mass $M_*\;(M_\odot)$", fontsize=18)
    ax1.set_xlim((x1lo, x1hi))
    ax1.tick_params(labelsize=16)
    ax1.legend(loc=1, fontsize=11)
    fig1.tight_layout()
    fig1.savefig("imgs/FigA1.png")

    ax2.hlines(1.0, xmin=x2lo, xmax=x2hi, linestyles="dashed", linewidth=1.0,
              colors="gray")
    ax2.set_ylabel("ASHC", fontsize=18)
    ax2.set_yscale("log")
    ax2.set_ylim((ylo, yhi))
    ax2.set_xlabel(r"$V-K$ Color ($\Delta$ mag)", fontsize=18)
    ax2.set_xlim((x2lo, x2hi))
    ax2.tick_params(labelsize=16)
    ax2.legend(loc=1, fontsize=11)
    fig2.tight_layout()
    fig2.savefig("imgs/Fig1.png")

    plt.show()


def plot_fig2(df: pd.DataFrame, show_names: bool=True, use_stmass: bool=False):

    # Load planet habitability and plotting data
    df = df[df["habitable"] == 1].reset_index()
    if show_names:
        df["plot_name"] = make_plot_names_pl(df, named_planets=NAMED_PLANETS)
    df = add_solar_system_planets(df, use_names=show_names)
    if show_names:
        df = _named_text_posn_fig4(df)

    # sort by pl_mass so smaller planets don't get covered
    df.sort_values(by="pl_bmasse", ascending=False, inplace=True)
    where_solarsys = df["plot_name"].isin(["Venus", "Earth", "Mars"])
    
    fig, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    ax4: plt.Axes
    cmap = cm.get_cmap("CMAP2" if use_stmass else "plasma_r")

    dots = np.log(df["pl_bmasse"])
    s_shift = 10
    s_scale = 4*35

    x = df["rperi"]
    y = df["Prot"]
    z = df["st_mass"] if use_stmass else df["VK_color"]
    s = _normalized_range(dots, dots, shift=s_shift, scale=s_scale)
    
    prot_lo = min(1e0, min(y))
    prot_hi = max(3e2, max(y))
    a_lo = min(1e-2, min(x))

    ax4.set_xlabel(r"a$_p$ (AU)", fontsize=18)
    ax4.set_xscale("log")
    ax4.set_xlim((a_lo, None))
    ax4.set_ylabel(r"P$_{rot}$ (d)", fontsize=18)
    ax4.set_yscale("log")
    ax4.set_ylim((prot_lo,prot_hi))
    ax4.tick_params(labelsize=16, size=5)

    # special plot for solar sys planets
    color_solarsys = cmap((np.array(z[where_solarsys])[0] - min(z)) / (max(z) - min(z)))
    ax4.scatter(x[where_solarsys], y[where_solarsys], color=color_solarsys, marker="o", s=s[where_solarsys], zorder=2.5, edgecolor="black")

    im = ax4.scatter(x[~where_solarsys], y[~where_solarsys], c=z[~where_solarsys], marker="o", s=s[~where_solarsys], cmap=cmap, zorder=2)
    cb = fig.colorbar(im, ax=ax4)
    cb.set_label(r"M$_*$ (M$_\odot$)" if use_stmass else r"V-K ($\Delta$mag)", fontsize=18)
    cb.ax.tick_params(labelsize=16)
    
    uncProt = 0.2
    npoints = 200
    protvals = np.linspace(prot_lo, prot_hi, num=npoints)
    if use_stmass:
        uncM = 0.05
        mvals = np.array([0.08 + 1e-5, 0.4, 0.8, 1.36 - 1e-5])
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
        vkvals = np.array([1.1 + 1e-5, 3.0, 5.0, 7.0 - 1e-5])
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
    plot_lines = ax4.plot(ravals, rpvals, label=labels, zorder=1)
    
    ax4.fill_betweenx(rpvals[:,0], ralower[:,0], raupper[:,0], alpha=0.25, zorder=1.3, color=c[0])
    ax4.fill_betweenx(rpvals[:,-1], ralower[:,-1], raupper[:,-1], alpha=0.25, zorder=1.4, color=c[-1])
    

    if show_names:
        where_named = df["plot_name"] != ""
        xtext = df.loc[where_named, "xtext"]
        ytext = df.loc[where_named, "ytext"]
        name = np.array(df.loc[where_named, "plot_name"])
        xy = np.stack([x[where_named], y[where_named]], axis=-1)
        xytext = np.stack([xtext, ytext], axis=-1)
        ct = xy.shape[0]
        kwargs_named = {"fontsize": 12, "zorder": 3}
        for i in range(ct):
            plt.annotate(text=name[i],
                         xy=xy[i],
                         xytext=xytext[i],
                         xycoords="data",
                         textcoords="offset points",
                         arrowprops=dict(arrowstyle="-",
                                         connectionstyle="arc3",
                                         color="gray"),
                         bbox=dict(pad=-2.5,
                                  facecolor="none",
                                  edgecolor="none"),
                         **kwargs_named)


    # how we break up the pl masses
    plmvals = np.log([0.107, 1.00, 17.2, 317.8])
    plmlabels = [r"Mars (0.107M$_\oplus)$", r"Earth (1.00M$_\oplus$)", r"Neptune (17.2M$_\oplus$)", r"Jupiter (318M$_\oplus$)"]
    csol = cmap((1.0 - min(z)) / (max(z)-min(z)))
    size_handles = [ax4.scatter([],[], color=csol, edgecolor="black",
            s=(_normalized_range(plmval, dots, shift=s_shift, scale=s_scale)), label=plmlabel) for plmlabel, plmval in zip(plmlabels, plmvals)]

    ax4.tick_params(labelsize=16, size=5)
    
    fig.tight_layout()
    
    legend_lines = ax4.legend(handles=plot_lines,
                              title=legtitle,
                              loc=(0.695,0.695),
                              title_fontsize=13,
                              fontsize=11)
    ax4.add_artist(legend_lines)
    ax4.legend(handles=size_handles, title="Planet Mass", loc=(0.62,0.01), title_fontsize=13, fontsize=11)
    fig.savefig("imgs/FigA2.png" if use_stmass else "imgs/Fig2.png")
    plt.show()



if __name__ == "__main__":

    # Load planet habitability and plotting data
    df = pd.read_csv('tables-merged/alfven_data.csv')
    df_h = df[df["habitable"] == 1].reset_index()

    # master plots
    plot_fig1(df, classcol="mass_class")
    plot_fig2(df, use_stmass=False)
    plot_fig2(df, use_stmass=True)