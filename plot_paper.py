import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


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
                 "TRAPPIST-1 f", "TRAPPIST-1 g"] # CHZ terran MHC>1
SOL_NAMES = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn",
             "Uranus", "Neptune"]


def plot_df(df: pd.DataFrame, ax: plt.Axes, xcol: str, ycol: str,
            errxcol: str="dRo", errycol: str="dMHC", logx=False, logy=False,
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
    vk_color_sol = 4.81 - 3.27  # Willmer 2018
    
    sol_data = {
        "hostname": ["Sol", "Sol", "Sol"],
        "pl_name": ["Venus", "Earth", "Mars"],
        "plot_name": ["Venus", "Earth", "Mars"],
        "Ro": [ro_sol, ro_sol, ro_sol],
        "e_Ro": [dro_sol, dro_sol, dro_sol],
        "MHC": [r_p_venus / ra_sol, r_p_earth / ra_sol, r_p_mars / ra_sol],
        "e_MHC": [dra_sol, dra_sol, dra_sol],
        "st_mass": [1., 1., 1.],
        "VK_color": [vk_color_sol, vk_color_sol, vk_color_sol],
        "mass_class": [1, 1, 0],
        "habitable": [1, 1, 1]
    }

    if use_names:
        sol_data.update({"plot_name": ["Venus", "Earth", "Mars"]})
        df["plot_name"] = df["plot_name"].apply(lambda x: str(int(x)) if isinstance(x, (float, int)) else x)

    sol = pd.DataFrame(sol_data)

    return pd.concat([df, sol], ignore_index=True)


if __name__ == "__main__":

    group = "mass_class"
    add_sol = True
    names_st = False
    
    colors = ["red", "green", "mediumorchid", "orange", "black"]
    color_h = "royalblue"

    # Load planet habitability and plotting data
    df = pd.read_csv('current-exo-data/alfven_data.csv')
    df_h = df[df["habitable"] == 1].reset_index()

    named_stars = df_h[(df_h[group] == 1) & (df_h["MHC"] > 1)]["hostname"].drop_duplicates().to_list()
    named_planets = df_h[(df_h[group] == 1) & (df_h["MHC"] > 1)]["pl_name"].to_list()
    named_objs = named_stars if names_st else named_planets

    plot_kwargs = {"alpha": 1.0, "s": 15, "lw_e": 1.0, "alpha_e": 0.25,
                   "s_h": 100, "color_h": color_h, "marker_h": "*", "label_h": f"CHZ ({len(df_h)})"}

    # master plots
    save_path_1 = "imgs/Fig1.png"
    fig, (ax1a, ax1b) = plt.subplots(nrows=2, ncols=1, sharey=True, figsize=(8,9))
    ax1a: plt.Axes
    ax1b: plt.Axes

    xlim_1a = (0, None)
    ylim_1a = (0, 75)
    save_path_1a = "imgs/Fig1a.png"
    plot_proc(df, "st_mass", "MHC", ax1a, group=group,
        xlim=xlim_1a, ylim=ylim_1a, named_objs=None, logy=True, plot_err=False,
        color_list=colors, reverse_grp=True, highlight_habitable=True,
        include_sol=True, leg_loc=1, **plot_kwargs)

    xlim_1b = (0.75, 7)
    ylim_1b = (0, 75)
    save_path_1b = "imgs/Fig1b.png"
    plot_proc(df, "VK_color", "MHC", ax1b, group=group,
        xlim=xlim_1b, ylim=ylim_1b, named_objs=None, logy=True, plot_err=False,
        color_list=colors, reverse_grp=True, highlight_habitable=True,
        include_sol=True, use_leg=False,**plot_kwargs)

    ax1a.annotate("a", xy=[0.02, 0.925], xycoords=ax1a.transAxes, fontsize=20)
    ax1b.annotate("b", xy=[0.02, 0.925], xycoords=ax1b.transAxes, fontsize=20)

    ax1a.set_xlabel(r"Stellar Mass $(M_\odot)$", fontsize=18)
    # ax1a.set_ylabel("MHC", fontsize=18)
    ax1a.tick_params(labelsize=16)
    ax1b.set_xlabel(r"$V-K_s$ Color ($\Delta$mag)", fontsize=18)
    ax1b.tick_params(labelsize=16)
    fig.supylabel("MHC", fontsize=18)

    fig.tight_layout()
    plt.savefig(save_path_1)
    plt.show()
    plt.close()


    fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    ax2: plt.Axes

    xlim_2 = (0.75, 7)
    ylim_2 = (0, 75)
    save_path_2 = "imgs/Fig2_color.png"
    plot_err = [True, True, True, True]
    plot_proc(df_h, "VK_color", "MHC", ax2, group=group, xlim=xlim_2,
        ylim=ylim_2, named_objs=named_objs, logy=True, plot_err=plot_err,
        names_st=names_st, color_list=colors, reverse_grp=True,
        highlight_habitable=False, include_sol=True, sort_col="MHC",
        leg_loc=1, errxcol="e_VK_color", errycol="e_MHC", **plot_kwargs)
    
    ax2.set_xlabel(r"$V-K_s$ Color ($\Delta$mag)", fontsize=18)
    ax2.set_ylabel("MHC", fontsize=18)
    ax2.tick_params(labelsize=16)
    fig.tight_layout()
    plt.savefig(save_path_2)
    plt.show()


    fig, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    ax3: plt.Axes

    ages = df.loc[df["MHC"].notnull() & df["st_age"].notnull(), "st_age"]
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
    ax3.set_title("Distribution of Stellar Age Estimates for Stars with an MHC")
    ax3.bar(x, hist, align="edge", width=-width, label=f"All Stars with Calulable MHC ({ages_count})")
    ax3.bar(width, ages_young, align="edge", width=-width, label=f"Young Stars <100Myr ({ages_young})")
    plt.legend()
    fig.tight_layout()
    plt.savefig("imgs/ages.png")
    
    # ages_errs = df.loc[df["MHC"].notnull(), "st_ageerr"]
    # hist, edges = ax3.hist(ages, yerr=ages_errs)
    # ax3.errorbar()
    plt.show()