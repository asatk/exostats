import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend_handler import HandlerPathCollection
import numpy as np
import pandas as pd

from alfven_estimates import ra_schrijver, taucM, taucVK
from matplotlib import cm

CMAP1 = LinearSegmentedColormap("CMAP1", dict(
    red=[(0.0, 0.0, 0.0),
         (0.5, 0.5, 0.5),
         (1.0, 1.0, 1.0)],
    green=[(0.0, 0.0, 0.0),
           (0.5, 0.5, 0.5),
           (1.0, 1.0, 1.0)],
    blue=[(0.0, 0.0, 0.0),
          (0.5, 0.5, 0.5),
          (1.0, 1.0, 1.0)]
))
CMAP2 = LinearSegmentedColormap.from_list("CMAP2", [
    "saddlebrown",
    "darkred",
    "red",
    "orange",
    "yellow"
])

cm.register_cmap(cmap=CMAP1)
cm.register_cmap(cmap=CMAP2)

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
                 "TRAPPIST-1 f", "TRAPPIST-1 g"] # CHZ terran ASHC>1
SOL_NAMES = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn",
             "Uranus", "Neptune"]


def plot_df(df: pd.DataFrame, ax: plt.Axes, xcol: str, ycol: str,
            errxcol: str="dRo", errycol: str="dASHC", logx=False, logy=False,
            show_names=False, plot_err=False, xnudge: float=0.005,
            ynudge: float=0.025, reverse_grp: bool=False,
            highlight_habitable: bool=False, color_h: str=COLOR_HABITABLE,
            marker_h: str="*", group: str="mass_class", color_list: list[str]=COLORS_1,
            class_list: list[str]=CLASS_LABELS, **kwargs):

    x = np.array(df[xcol])
    y = np.array(df[ycol])
    grp_num = df[group].iat[0]
    color = kwargs.get("color", color_list[grp_num])
    label = kwargs.get("label", f"{class_list[grp_num]} ({len(df)})")
    len_h = len(df[df["habitable"] == 1])
    label_h = kwargs.get("label_h", f"CHZ ({len_h})")
    alpha = kwargs.get("alpha", 1.0)
    alpha_h = kwargs.get("alpha_h", 1.0)
    alpha_e = kwargs.get("alpha_e", 0.4)
    s = kwargs.get("s", 20)
    s_h = kwargs.get("s_h", 100)
    lw = kwargs.get("lw", 0)
    lw_h = kwargs.get("lw_h", 0)
    lw_e = kwargs.get("lw_e", 1.1)

    if isinstance(plot_err, list):
        plot_err = plot_err[grp_num]
    
    if logx:
        ax.set_xscale("log")

    if logy:
        ax.set_yscale("log")
    
    if plot_err:
        xerr = np.array(df[errxcol])
        yerr = np.array(df[errycol])
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, linestyle="None",
            ecolor="#333333", elinewidth=lw_e, alpha=alpha_e, zorder=0.0)

    if show_names:
        xtext = x + xnudge
        ytext = (y * (1 + ynudge)) if logy else (y + ynudge)
        name = df["plot_name"]
        ct = x.shape[0]
        kwargs = ct * [{"fontsize": 12, "zorder": 10.0}]
        list(map(plt.text, xtext, ytext, name, kwargs))

    scatter_zorder = 5 - (grp_num if reverse_grp else -grp_num)
    if highlight_habitable:
        h = np.array(df["habitable"] == 1)
        x_h = x[h]
        y_h = y[h]
        nh = np.logical_not(h)
        x_nh = x[nh]
        y_nh = y[nh]
        ax.scatter(x_h, y_h, c=color_h, alpha=alpha_h, marker=marker_h,
                   zorder=9.5, s=s_h, label=label_h, lw=lw_h)
        ax.scatter(x_nh, y_nh, color=color, alpha=alpha, zorder=scatter_zorder,
                   s=s, label=label, lw=lw)
    else:
        ax.scatter(x, y, color=color, alpha=alpha, label=label,
                   zorder=scatter_zorder, s=s, lw=lw)


def _make_name(row: pd.DataFrame) -> str:
    
    s = str(row["grp_num"])
    if row["grp_len"] == 1:
        return s
    
    s += row["pl_letter"]
    return s


def make_plot_names_st(df: pd.DataFrame, named_stars: list[str]=NAMED_STARS,
                    sort_col: str="hostname") -> pd.DataFrame:
    
    if len(NAMED_STARS) == 0:
        df_all_plot_name = pd.Series(["" for _ in df.iterrows()])
        return df_all_plot_name
    
    hostnames = pd.Series(named_stars, name="hostname")
    
    df_fltr = pd.merge(df, hostnames, how="inner", on="hostname").sort_values(
        by=sort_col)[["hostname", "pl_name", "pl_letter"]]
    df_fltr["grp_num"] = df_fltr.groupby("hostname").ngroup() + 1
    # df_map = df_fltr.drop_duplicates(subset="hostname")[["grp_num", "hostname"]]
    
    df_grp_len = df_fltr.groupby("hostname")["grp_num"].count()
    df_grp_len.rename("grp_len", inplace=True)
    df_grps = pd.merge(df_fltr, df_grp_len, how="left", on="hostname")
    
    df_grps["plot_name"] = df_grps.apply(_make_name, axis=1)
    df_all_plot_name = pd.merge(df, df_grps, how="left", on="pl_name")["plot_name"]
    df_all_plot_name.fillna("", inplace=True)

    return df_all_plot_name


def make_plot_names_pl(df: pd.DataFrame, named_planets: list[str]=NAMED_PLANETS,
                    sort_col: str="pl_name") -> pd.DataFrame:
    
    pl_names = pd.Series(named_planets, name="pl_name")

    df_fltr = pd.merge(df, pl_names, how="inner", on="pl_name").sort_values(by=sort_col)[["pl_name"]]
    df_fltr["plot_name"] = np.asarray(df_fltr.groupby("pl_name", sort=False).ngroup() + 1, dtype=str)
    df_map = df_fltr[["plot_name", "pl_name"]]

    df_all_plot_name = pd.merge(df, df_map, how="left", on="pl_name")["plot_name"]
    df_all_plot_name.fillna("", inplace=True)

    return df_all_plot_name


def plot_proc(df: pd.DataFrame, xcol: str, ycol: str, ax: plt.Axes,
              group: str=None, xlim: tuple[float]=None,
              ylim: tuple[float]=None, named_objs: list[str]=None,
              names_st: bool=True, logx: bool=False,
              logy: bool=False, use_leg: bool=True, leg_loc: int=4, include_sol: bool=False,
              **kwargs) -> None:

    df = df.copy()

    show_names = named_objs is not None

    if show_names:
        sort_col = kwargs.get("sort_col", "pl_name")
        if names_st:
            df["plot_name"] = \
                make_plot_names_st(df, named_stars=named_objs, sort_col=sort_col)
        else:
            df["plot_name"] = \
                make_plot_names_pl(df, named_planets=named_objs, sort_col=sort_col)

    x = np.array(df[xcol])
    y = np.array(df[ycol])

    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    xspan = xmax - xmin
    ymin = np.nanmin(y)
    ymax = np.nanmax(y)
    yspan = ymax - ymin
    pad = 0.01
    pad_log = 0.75

    if xlim is not None:
        xlimmin = xlim[0]
        xlimmax = xlim[1]
        if xlimmin == 0 and logx:
            xlimmin = xmin * pad_log
        if xlimmin is None:
            xlimmin = xmin - pad * xspan
        if xlimmax is None:
            xlimmax = xmax + pad * xspan
    else:
        if logx:
            xlimmin = xmin * pad_log
        else:
            xlimmin = xmin - pad * xspan
        xlimmax = xmax + pad * xspan
    ax.set_xlim((xlimmin, xlimmax))

    if ylim is not None:
        ylimmin = ylim[0]
        ylimmax = ylim[1]
        if ylimmin == 0 and logy:
            ylimmin = ymin * pad_log
        if ylimmin is None:
            ylimmin = ymin - pad * yspan
        if ylimmax is None:
            ylimmax = ymax + pad * yspan
    else:
        ylimmin = ymin - pad * (ymax - ymin)
        ylimmax = ymax + pad * (ymax - ymin)
    ax.set_ylim((ylimmin, ylimmax))

    ax.hlines(1.0, xmin=xlimmin, xmax=xlimmax, linestyles="dashed", linewidth=1.0,
              colors="gray")
    
    # just for the counts
    df_subset = df[(df[xcol] <= xlimmax) & (df[xcol] >= xlimmin) & (df[ycol] <= ylimmax) & (df[ycol] >= ylimmin)]
    count_subset = df_subset.loc[df_subset[xcol].notnull() & df_subset[ycol].notnull(), "pl_name"].count()
    count_input = df.loc[df[xcol].notnull() & df[ycol].notnull(), "pl_name"].count()
    print(f"Number of rows (NOT incl. 3 SolSys pls) in range {xlimmin:.3f} <= {xcol} <= {xlimmax:.3f}; {ylimmin:.3f} <= {ycol} <= {ylimmax:.3f}: {count_subset}/{count_input}")

    if include_sol:
        df = add_solar_system_planets(df, use_names=show_names)
        df_subset = df[(df[xcol] <= xlimmax) & (df[xcol] >= xlimmin) & (df[ycol] <= ylimmax) & (df[ycol] >= ylimmin)]

    if group is not None:
        df_plot = df_subset.groupby(group, as_index=False)
    else:
        df_plot = df_subset

    if group is not None:
        df_plot.apply(plot_df, ax=ax, xcol=xcol, ycol=ycol, show_names=show_names, logx=logx, logy=logy, group=group, **kwargs)
    else:
        plot_df(df=df_plot, ax=ax, xcol=xcol, ycol=ycol, show_names=show_names, logx=logx, logy=logy, **kwargs)
    
    if use_leg:
        hand, labl = ax.get_legend_handles_labels()
        _, index_u = np.unique(labl, return_index=True)
        labl_u = np.array(labl)[index_u]
        hand_u = np.array(hand)[index_u]

        len_labl_u = len(labl_u)
        # reverse if reverse_grp?
        if len_labl_u == 4:
            ind_shuffle = [1,3,2,0]
        elif len_labl_u == 6:
            ind_shuffle = [3,5,4,1,2,0]
        else:
            ind_shuffle = np.arange(0, len_labl_u)

        labl_u = labl_u[ind_shuffle]
        hand_u = hand_u[ind_shuffle]

        # find better way to match the order of legend labels by matching w labels list idx
        ax.legend(hand_u, labl_u, loc=leg_loc)


def add_solar_system_planets(df: pd.DataFrame, use_names: bool=False) -> pd.DataFrame:
    ro_sol = 1.85
    dro_sol = 0.26
    # ra_sol = 0.1383 # 695700km in AU
    ra_sol = 0.0930
    dra_sol = 0.02  # 20% error (cycle variation)
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


# just do a simple plot(df - df_h) empty markers + plot(df_h) filled markers
def plot_old_fig1(df: pd.DataFrame):

    group = "mass_class"
    
    colors = ["red", "green", "mediumorchid", "orange", "black"]
    color_h = "royalblue"

    # Load planet habitability and plotting data
    df_h = df[df["habitable"] == 1]

    plot_kwargs = {"alpha": 1.0, "s": 15, "lw_e": 1.0, "alpha_e": 0.25,
                   "s_h": 100, "color_h": color_h, "marker_h": "*", "label_h": f"CHZ ({len(df_h)})"}

    save_path_1 = "imgs/Fig1.png"
    # fig, (ax1a, ax1b) = plt.subplots(nrows=2, ncols=1, sharey=True, figsize=(8,9))
    fig, (ax1a, ax1b) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(16,6))
    ax1a: plt.Axes
    ax1b: plt.Axes

    xlim_1a = (0, None)
    ylim_1a = (0, 75)
    # save_path_1a = "imgs/Fig1a.png"
    plot_proc(df, "st_mass", "ASHC", ax1a, group=group,
        xlim=xlim_1a, ylim=ylim_1a, named_objs=None, logy=True, plot_err=False,
        color_list=colors, reverse_grp=True, highlight_habitable=True,
        include_sol=True, leg_loc=1, **plot_kwargs)

    xlim_1b = (0.75, 7)
    ylim_1b = (0, 75)
    # save_path_1b = "imgs/Fig1b.png"
    plot_proc(df, "VK_color", "ASHC", ax1b, group=group,
        xlim=xlim_1b, ylim=ylim_1b, named_objs=None, logy=True, plot_err=False,
        color_list=colors, reverse_grp=True, highlight_habitable=True,
        include_sol=True, use_leg=False,**plot_kwargs)

    ax1a.annotate("a", xy=[0.02, 0.925], xycoords=ax1a.transAxes, fontsize=20)
    ax1b.annotate("b", xy=[0.02, 0.925], xycoords=ax1b.transAxes, fontsize=20)

    ax1a.set_xlabel(r"Stellar Mass $(M_\odot)$", fontsize=18)
    # ax1a.set_ylabel("ASHC", fontsize=18)
    ax1a.tick_params(labelsize=16)
    ax1b.set_xlabel(r"$V-K_s$ Color ($\Delta$mag)", fontsize=18)
    ax1b.tick_params(labelsize=16)
    fig.supylabel("ASHC", fontsize=18)

    fig.tight_layout()
    plt.savefig(save_path_1)
    # plt.show()
    plt.close()


def plot_fig2(df: pd.DataFrame):
    group = "mass_class"
    add_sol = True
    names_st = False
    
    colors = ["red", "green", "mediumorchid", "orange", "black"]
    color_h = "royalblue"

    # Load planet habitability and plotting data
    df_h = df[df["habitable"] == 1].reset_index()

    named_stars = df_h[(df_h[group] == 1) & (df_h["ASHC"] > 1)]["hostname"].drop_duplicates().to_list()
    named_planets = df_h[(df_h[group] == 1) & (df_h["ASHC"] > 1)]["pl_name"].to_list()
    named_objs = named_stars if names_st else named_planets

    plot_kwargs = {"alpha": 1.0, "s": 15, "lw_e": 1.0, "alpha_e": 0.25,
                   "s_h": 100, "color_h": color_h, "marker_h": "*", "label_h": f"CHZ ({len(df_h)})"}
    
    fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    ax2: plt.Axes

    xlim_2 = (0.75, 7)
    ylim_2 = (0, 75)
    save_path_2 = "imgs/Fig2_color.png"
    plot_err = [True, True, True, True]
    plot_proc(df_h, "VK_color", "ASHC", ax2, group=group, xlim=xlim_2,
        ylim=ylim_2, named_objs=named_objs, logy=True, plot_err=plot_err,
        names_st=names_st, color_list=colors, reverse_grp=True,
        highlight_habitable=False, include_sol=True, sort_col="ASHC",
        leg_loc=1, errxcol="e_VK_color", errycol="e_ASHC", **plot_kwargs)
    
    ax2.set_xlabel(r"$V-K_s$ Color ($\Delta$mag)", fontsize=18)
    ax2.set_ylabel("ASHC", fontsize=18)
    ax2.tick_params(labelsize=16)
    fig.tight_layout()
    plt.savefig(save_path_2)
    # plt.show()
    plt.close()


def plot_fig3(df: pd.DataFrame):
    group = "mass_class"
    add_sol = True
    names_st = False
    
    colors = ["red", "green", "mediumorchid", "orange", "black"]
    color_h = "royalblue"

    # Load planet habitability and plotting data
    df_h = df[df["habitable"] == 1].reset_index()

    named_stars = df_h[(df_h[group] == 1) & (df_h["ASHC"] > 1)]["hostname"].drop_duplicates().to_list()
    named_planets = df_h[(df_h[group] == 1) & (df_h["ASHC"] > 1)]["pl_name"].to_list()
    named_objs = named_stars if names_st else named_planets

    plot_kwargs = {"alpha": 1.0, "s": 15, "lw_e": 1.0, "alpha_e": 0.25,
                   "s_h": 100, "color_h": color_h, "marker_h": "*", "label_h": f"CHZ ({len(df_h)})"}
    
    fig, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    ax3: plt.Axes

    df_st = df.drop_duplicates(subset="hostname")
    ages = df.loc[df["Prot"].notnull() & df["st_age"].notnull(), "st_age"]
    ages_young = ages[ages < 0.1].count()
    ages_count = ages.count()
    ages_max = np.max(ages)
    bins = 10
    width = ages_max / bins
    hist, edges = np.histogram(ages, bins=bins)
    # x = (edges[1:] + edges[:-1])/2
    x = edges[1:]
    ax3.set_xlabel("Stellar Age (Gyr)")
    ax3.set_xlim((0, ages_max))
    ax3.set_ylabel("Counts")
    ax3.set_title(r"Distribution of Stellar Age Estimates for Stars with a $P_{rot}")
    ax3.bar(x, hist, align="edge", width=-width, label=f"All Stars with Measured $P_{{rot}}$ ({ages_count})")
    ax3.bar(width, ages_young, align="edge", width=-width, label=f"Young Stars <100Myr ({ages_young})")
    plt.legend()
    fig.tight_layout()
    plt.savefig("imgs/ages.png")
    # plt.show()
    plt.close()


def _normalized_range(data: np.array, data_range: np.array, shift: float=0., scale: float=1.):
    return shift + scale * (data - np.amin(data_range)) / (np.amax(data_range) - np.amin(data_range))


def _named_text_posn_fig4(df: pd.DataFrame):
    
    df["xtext"] = 10
    df["ytext"] = 10
    
    df.loc[df["plot_name"] == "1", "xtext"] = 7
    df.loc[df["plot_name"] == "1", "ytext"] = -13

    df.loc[df["plot_name"] == "3", "xtext"] = 5
    df.loc[df["plot_name"] == "3", "ytext"] = 15

    df.loc[df["plot_name"] == "4", "xtext"] = 7
    df.loc[df["plot_name"] == "4", "ytext"] = -13

    df.loc[df["plot_name"] == "5", "xtext"] = 9
    df.loc[df["plot_name"] == "5", "ytext"] = -11

    df.loc[df["plot_name"] == "6", "xtext"] = 15
    df.loc[df["plot_name"] == "6", "ytext"] = 5

    df.loc[df["plot_name"] == "9", "xtext"] = 17
    df.loc[df["plot_name"] == "9", "ytext"] = 3

    df.loc[df["plot_name"] == "10", "xtext"] = 19
    df.loc[df["plot_name"] == "10", "ytext"] = 1
    
    df.loc[df["plot_name"] == "Venus", "xtext"] = 5
    df.loc[df["plot_name"] == "Venus", "ytext"] = -15
    
    df.loc[df["plot_name"] == "Mars", "xtext"] = 13
    df.loc[df["plot_name"] == "Mars", "ytext"] = -7

    df.loc[df["plot_name"] == "Earth", "xtext"] = 15
    df.loc[df["plot_name"] == "Earth", "ytext"] = 5

    return df


def plot_fig1(df: pd.DataFrame, classcol: str="mass_class", use_stmass: bool=False):
    
    df = df.copy(deep=False)
    colors = ["red", "green", "mediumorchid", "orange", "black"]
    grp_counts = df.groupby(by=classcol).count()["ASHC"]
    grp_names = ["subterran", "terran", "superterran", "giant", "no class"]

    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    ax1: plt.Axes
    ax2: plt.Axes
    
    df_nh = df[df["habitable"] == 0].reset_index(inplace=False)
    df_u, df_inv = np.unique(df_nh["mass_class"], return_inverse=True, axis=0)

    alpha_nh = 0.66
    for idx in reversed(range(max(df_inv) + 1)):
        grp = df_u[idx]
        label = f"{grp_names[idx]} ({grp_counts.at[grp]})"
        temp = df_nh[df_inv == idx]
        
        ax1.plot(np.array(temp["st_mass"]), np.array(temp["ASHC"]),
                 c=colors[grp], linestyle="none", marker="+", fillstyle="none", alpha=alpha_nh, label=label)
        ax2.plot(np.array(temp["VK_color"]), np.array(temp["ASHC"]),
                 c=colors[grp], linestyle="none", marker="+", fillstyle="none", alpha=alpha_nh, label=label)

    df_h = df[df["habitable"] == 1].reset_index(inplace=False)
    df_u, df_inv = np.unique(df_h["mass_class"], return_inverse=True, axis=0)

    for idx in reversed(range(max(df_inv) + 1)):
        grp = df_u[idx]
        temp = df_h[df_inv == idx]
        ax1.plot(np.array(temp["st_mass"]), np.array(temp["ASHC"]),
                 c=colors[grp], linestyle="none", marker="o", fillstyle="full", alpha=1.0)
        ax2.plot(np.array(temp["VK_color"]), np.array(temp["ASHC"]),
                 c=colors[grp], linestyle="none", marker="o", fillstyle="full", alpha=1.0)

    x1lo = 0.0
    x1hi = 2.6
    x2lo = 0.5
    x2hi = 8.0
    ylo = 5e-2
    yhi = 100

    print(len(df[(df["ASHC"] >= ylo) &
                 (df["ASHC"] <= yhi) &
                 (df["st_mass"] >= x1lo) &
                 (df["st_mass"] <= x1hi)]))
    
    print(len(df[(df["ASHC"] >= ylo) &
                 (df["ASHC"] <= yhi) &
                 (df["VK_color"] >= x2lo) &
                 (df["VK_color"] <= x2hi)]))
    
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
    # plt.savefig("imgs/Fig1.png")

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
    # plt.savefig("imgs/FigA1.png")

    plt.show()


def plot_fig2(df: pd.DataFrame, show_names: bool=True, use_stmass: bool=True):

    # Load planet habitability and plotting data
    df = df[df["habitable"] == 1].reset_index()
    if show_names:
        df["plot_name"] = make_plot_names_pl(df, named_planets=NAMED_PLANETS, sort_col="pl_name")
    df = add_solar_system_planets(df, use_names=show_names)
    if show_names:
        df = _named_text_posn_fig4(df)

    # sort by pl_mass so smaller planets don't get covered
    df.sort_values(by="pl_bmasse", ascending=False, inplace=True)
    where_solarsys = df["plot_name"].isin(["Venus", "Earth", "Mars"])
    
    fig, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    ax4: plt.Axes
    cmap = cm.get_cmap("CMAP2" if use_stmass else "viridis_r")

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

    ax4.set_xlabel("a$_p$ (AU)", fontsize=18)
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
    cb.set_label("M$_*$ (M$_\odot$)" if use_stmass else "V-K ($\Delta$mag)", fontsize=18)
    cb.ax.tick_params(labelsize=16)
    

    npoints = 100
    protvals = np.linspace(prot_lo, prot_hi, num=npoints)
    if use_stmass:
        mvals = pd.Series([0.25, 0.5, 0.75, 1.0, 1.25])
        rovals = np.outer(1/taucM(mvals), protvals).T
        ravals = ra_schrijver(rovals)
        rpvals = np.repeat(protvals, repeats=len(mvals), axis=0).reshape((npoints,len(mvals)))
        labels = [f"$M_*=${i:.02f}M$_\odot$" for i in mvals]
        legtitle = "Extent of Mean\n" + r"Alfv$\'e$n Surface"
        c = [cmap((mval - min(z)) / (max(z)-min(z))) for mval in mvals]
    else:
        vkvals = pd.Series([1.5, 3.0, 4.5, 6.0])
        rovals = np.outer(1/taucVK(vkvals), protvals).T
        ravals = ra_schrijver(rovals)
        rpvals = np.repeat(protvals, repeats=len(vkvals), axis=0).reshape((npoints,len(vkvals)))
        labels = [f"$V-K=${i:.02f}" for i in vkvals]
        legtitle = "Extent of Mean\n" + r"Alfv$\'e$n Surface"
        c = [cmap((vkval - min(z)) / (max(z)-min(z))) for vkval in vkvals]
    
    ax4.set_prop_cycle("color", c)
    plot_lines = ax4.plot(ravals, rpvals, label=labels, zorder=1)
    

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
    plmlabels = ["Mars (0.107M$_\oplus)$", "Earth (1.00M$_\oplus$)", "Neptune (17.2M$_\oplus$)", "Jupiter (318M$_\oplus$)"]
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
    fig.savefig("imgs/Fig2_stmass.png" if use_stmass else "imgs/Fig2_vkcolor.png")
    plt.show()


def plot_figA2_sm(df: pd.DataFrame, show_names: bool=True):

    # Load planet habitability and plotting data
    df = df[(df["habitable"] == 1) & np.isfinite(df["RoM"])].reset_index()
    if show_names:
        df["plot_name"] = make_plot_names_pl(df, named_planets=NAMED_PLANETS, sort_col="pl_name")
    df = add_solar_system_planets(df, use_names=show_names)
    if show_names:
        df = _named_text_posn_fig4(df)

    # sort by pl_mass so smaller planets don't get covered
    df.sort_values(by="pl_bmasse", ascending=False, inplace=True)
    where_solarsys = df["plot_name"].isin(["Venus", "Earth", "Mars"])
    
    fig, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    ax4: plt.Axes
    cmap = cm.get_cmap("CMAP2")

    dots = np.log(df["pl_bmasse"])
    s_shift = 10
    s_scale = 4*35

    x = df["rperi"]
    y = df["Prot"]
    z = df["st_mass"]
    s = _normalized_range(dots, dots, shift=s_shift, scale=s_scale)
    
    prot_lo = min(1e0, min(y))
    prot_hi = max(3e2, max(y))
    a_lo = min(1e-2, min(x))

    ax4.set_xlabel("a$_p$ (AU)", fontsize=18)
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
    cb.set_label("M$_*$ (M$_\odot$)", fontsize=18)
    cb.ax.tick_params(labelsize=16)
    

    npoints = 100
    protvals = np.linspace(prot_lo, prot_hi, num=npoints)
    
    mvals = pd.Series([0.25, 0.5, 0.75, 1.0, 1.25])
    rovals = np.outer(1/taucM(mvals), protvals).T
    ravals = ra_schrijver(rovals)
    rpvals = np.repeat(protvals, repeats=len(mvals), axis=0).reshape((npoints,len(mvals)))
    labels = [f"$M_*=${i:.02f}M$_\odot$" for i in mvals]
    legtitle = "Extent of Mean\n" + r"Alfv$\'e$n Surface"
    c = [cmap((mval - min(z)) / (max(z)-min(z))) for mval in mvals]
    
    ax4.set_prop_cycle("color", c)
    plot_lines = ax4.plot(ravals, rpvals, label=labels, zorder=1)
    

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
    plmlabels = ["Mars (0.107M$_\oplus)$", "Earth (1.00M$_\oplus$)", "Neptune (17.2M$_\oplus$)", "Jupiter (318M$_\oplus$)"]
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
    fig.savefig("imgs/FigA1.png")
    plt.show()


def plot_figA2_vk(df: pd.DataFrame, show_names: bool=True):

    # Load planet habitability and plotting data
    df = df[(df["habitable"] == 1) & np.isfinite(df["RoVK"])].reset_index()
    if show_names:
        df["plot_name"] = make_plot_names_pl(df, named_planets=NAMED_PLANETS, sort_col="pl_name")
    df = add_solar_system_planets(df, use_names=show_names)
    if show_names:
        df = _named_text_posn_fig4(df)

    # sort by pl_mass so smaller planets don't get covered
    df.sort_values(by="pl_bmasse", ascending=False, inplace=True)
    where_solarsys = df["plot_name"].isin(["Venus", "Earth", "Mars"])
    
    fig, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    ax4: plt.Axes
    cmap = cm.get_cmap("viridis_r")

    dots = np.log(df["pl_bmasse"])
    s_shift = 10
    s_scale = 4*35

    x = df["rperi"]
    y = df["Prot"]
    z = df["VK_color"]
    s = _normalized_range(dots, dots, shift=s_shift, scale=s_scale)
    
    prot_lo = min(1e0, min(y))
    prot_hi = max(3e2, max(y))
    a_lo = min(1e-2, min(x))

    ax4.set_xlabel("a$_p$ (AU)", fontsize=18)
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
    cb.set_label("V-K ($\Delta$mag)", fontsize=18)
    cb.ax.tick_params(labelsize=16)
    

    npoints = 100
    protvals = np.linspace(prot_lo, prot_hi, num=npoints)
    
    vkvals = pd.Series([1.5, 3.0, 4.5, 6.0])
    rovals = np.outer(1/taucVK(vkvals), protvals).T
    ravals = ra_schrijver(rovals)
    rpvals = np.repeat(protvals, repeats=len(vkvals), axis=0).reshape((npoints,len(vkvals)))
    labels = [f"$V-K=${i:.02f}" for i in vkvals]
    legtitle = "Extent of Mean\n" + r"Alfv$\'e$n Surface"
    c = [cmap((vkval - min(z)) / (max(z)-min(z))) for vkval in vkvals]
    
    ax4.set_prop_cycle("color", c)
    plot_lines = ax4.plot(ravals, rpvals, label=labels, zorder=1)
    

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
    plmlabels = ["Mars (0.107M$_\oplus)$", "Earth (1.00M$_\oplus$)", "Neptune (17.2M$_\oplus$)", "Jupiter (318M$_\oplus$)"]
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
    fig.savefig("imgs/FigA2.png")
    plt.show()


if __name__ == "__main__":

    # Load planet habitability and plotting data
    df = pd.read_csv('current-exo-data/alfven_data.csv')
    df_h = df[df["habitable"] == 1].reset_index()

    # master plots
    plot_fig1(df, classcol="mass_class")
    # plot_fig2(df, use_stmass=True)
    # plot_fig2(df, use_stmass=False)
    # plot_figA2_st(df)
    # plot_figA2_vk(df)