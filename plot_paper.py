from matplotlib import pyplot as plt
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
            errxcol: str="dRo", errycol: str="dMHC",logy=False, show_names=False,
            plot_err=False, xnudge: float=0.05, ynudge: float=0.05,
            reverse_grp: bool=False, highlight_habitable: bool=False,
            color_h: str=COLOR_HABITABLE, marker_h: str="*",
            color_list: list[str]=COLORS_1,
            class_list: list[str]=CLASS_LABELS, **kwargs):

    x = np.array(df[xcol])
    y = np.array(df[ycol])
    grp_num = df.mass_class.iat[0]  # TODO >>>> MUST CHANGE THIS
    color = color_list[grp_num]
    label = class_list[grp_num]
    alpha = kwargs.get("alpha", 1.0)
    alpha_h = kwargs.get("alpha_h", 1.0)
    alpha_e = kwargs.get("alpha_e", 0.4)
    s = kwargs.get("s", 20)
    s_h = kwargs.get("s_h", 100)
    lw = kwargs.get("lw", 0)
    lw_h = kwargs.get("lw_h", 0)
    lw_e = kwargs.get("lw_e", 1.1)
    xlabel = kwargs.get("xlabel", "Rossby Number")
    ylabel = kwargs.get("ylabel", "MHC")

    if isinstance(plot_err, list):
        plot_err = plot_err[grp_num]
    
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
        kwargs = ct * [{"fontsize": 10, "zorder": 10.0}]
        list(map(plt.text, xtext, ytext, name, kwargs))

    scatter_zorder = 5 - (grp_num if reverse_grp else -grp_num)
    if highlight_habitable:
        h = np.array(df["habitable"] == 1)
        x_h = x[h]
        y_h = y[h]
        nh = np.logical_not(h)
        x_nh = x[nh]
        y_nh = y[nh]
        ax.scatter(x_h, y_h, color=color_h, alpha=alpha_h, marker=marker_h,
                   zorder=9.5, s=s_h, label="CHZ", lw=lw_h)
        ax.scatter(x_nh, y_nh, color=color, alpha=alpha, zorder=scatter_zorder,
                   s=s, label=label, lw=lw)
    else:
        ax.scatter(x, y, color=color, alpha=alpha, label=label,
                   zorder=scatter_zorder, s=s, lw=lw)
    
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(labelsize=16)


def _make_name(row: pd.DataFrame) -> str:
    
    s = str(row["grp_num"])
    if row["grp_len"] == 1:
        return s
    
    s += row["pl_letter"]
    return s


def make_plot_names_st(df: pd.DataFrame, return_mapping: bool=False,
                    named_stars: list[str]=NAMED_STARS,
                    sort_col: str="hostname") -> pd.DataFrame:
    
    if len(NAMED_STARS) == 0:
        df_all_plot_name = pd.Series(["" for _ in df.iterrows()])
        if return_mapping:
            return df_all_plot_name, {}
        return df_all_plot_name
    
    hostnames = pd.Series(named_stars, name="hostname")
    
    df_fltr = pd.merge(df, hostnames, how="inner", on="hostname").sort_values(
        by=sort_col)[["hostname", "pl_name", "pl_letter"]]
    df_fltr["grp_num"] = df_fltr.groupby("hostname").ngroup() + 1
    df_map = df_fltr.drop_duplicates(subset="hostname")[["grp_num", "hostname"]]
    name_mapping = df_map.to_dict(orient="split")["data"]
    
    df_grp_len = df_fltr.groupby("hostname")["grp_num"].count()
    df_grp_len.rename("grp_len", inplace=True)
    df_grps = pd.merge(df_fltr, df_grp_len, how="left", on="hostname")
    
    df_grps["plot_name"] = df_grps.apply(_make_name, axis=1)
    df_all_plot_name = pd.merge(df, df_grps, how="left", on="pl_name")["plot_name"]
    df_all_plot_name.fillna("", inplace=True)

    if return_mapping:
        return df_all_plot_name, name_mapping
    return df_all_plot_name


def make_plot_names_pl(df: pd.DataFrame, return_mapping: bool=False,
                    named_planets: list[str]=NAMED_PLANETS,
                    sort_col: str="pl_name") -> pd.DataFrame:
    
    pl_names = pd.Series(named_planets, name="pl_name")

    df_fltr = pd.merge(df, pl_names, how="inner", on="pl_name").sort_values(by=sort_col)[["pl_name"]]
    df_fltr["plot_name"] = np.asarray(df_fltr.groupby("pl_name", sort=False).ngroup() + 1, dtype=int)
    df_map = df_fltr[["plot_name", "pl_name"]]
    name_mapping = df_map.to_dict(orient="split")["data"]

    df_all_plot_name = pd.merge(df, df_map, how="left", on="pl_name")["plot_name"]
    df_all_plot_name.fillna("", inplace=True)

    if return_mapping:
        return df_all_plot_name, name_mapping
    return df_all_plot_name


def plot_proc(df: pd.DataFrame, xcol: str, ycol: str, group: str=None,
              save_path: str=None, xlim: tuple[float]=None,
              ylim: tuple[float]=None, named_objs: list[str]=None,
              names_table: bool=False, names_st: bool=True, logy: bool=False,
              bottom_adj: float=0.125, left_adj: float=0.08, use_adjust: bool=False,
              leg_loc: int=4, include_sol: bool=False, **kwargs) -> None:

    df = df.copy()

    show_names = named_objs is not None

    if show_names:
        sort_col = kwargs.get("sort_col", "pl_name")
        if names_table:
            if names_st:
                df["plot_name"], name_mapping = \
                    make_plot_names_st(df, return_mapping=True,
                        named_stars=named_objs, sort_col=sort_col)
            else:
                df["plot_name"], name_mapping = \
                    make_plot_names_pl(df, return_mapping=True,
                        named_planets=named_objs, sort_col=sort_col)
        else:
            if names_st:
                df["plot_name"] = \
                    make_plot_names_st(df, return_mapping=False,
                        named_stars=named_objs, sort_col=sort_col)
            else:
                df["plot_name"] = \
                    make_plot_names_pl(df, return_mapping=False,
                        named_planets=named_objs, sort_col=sort_col)
    else:
        names_table = False

    if include_sol:
        df = add_solar_system_planets(df, use_names=show_names)

    if group is not None:
        df_plot = df.groupby(group, as_index=False)
    else:
        df_plot = df

    ax1: plt.Axes
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    
    x = np.array(df[xcol])
    y = np.array(df[ycol])

    if xlim is not None:
        xmin = xlim[0]
        xmax = xlim[1]
        if xmin is None:
            xmin = 0
        if xmax is None:
            xmax = np.max(x) * 1.02
    else:
        xmin = 0
        xmax = np.max(x) * 1.02
    ax1.set_xlim((xmin, xmax))

    if ylim is not None:
        ymin = ylim[0]
        ymax = ylim[1]
        if ymin == 0.0 and logy:
            ymin = np.min(y) * 0.75
        if ymin is None:
            ymin = np.min(y) * 0.75
        if ymax is None:
            ymax = np.max(y) * 1.25
    else:
        ymin = np.min(y) * 0.75
        ymax = np.max(y) * 1.25
    ax1.set_ylim((ymin, ymax))

    ax1.hlines(1.0, xmin=xmin, xmax=xmax, linestyles="dashed", linewidth=1.0,
              colors="gray")

    if group is not None:
        df_plot.apply(plot_df, ax=ax1, xcol=xcol, ycol=ycol, show_names=show_names, logy=logy, **kwargs)
    else:
        plot_df(df=df_plot, ax=ax1, xcol=xcol, ycol=ycol, show_names=show_names, logy=logy, **kwargs)
    
    hand, labl = ax1.get_legend_handles_labels()
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
    ax1.legend(hand_u, labl_u, loc=leg_loc)

    if names_table:
        name_mapping = sorted(name_mapping, key=lambda x: x[0])
        ax1.table(cellText=name_mapping, colWidths=[0.05, 0.15], loc="right")

    plt.tight_layout()
    if use_adjust:
        plt.subplots_adjust(bottom=bottom_adj, left=left_adj)
    if save_path is not None:
        fig.savefig(save_path)
    
    plt.show()
    plt.close()


def add_solar_system_planets(df: pd.DataFrame, use_names: bool=False) -> pd.DataFrame:
    ro_sol = 1.85
    dro_sol = 0.26
    ra_sol = 0.1
    dra_sol = 0.02
    r_p_venus = 0.718
    r_p_earth = 0.983
    r_p_mars = 1.381
    
    sol_data = {
        "hostname": ["Sol", "Sol", "Sol"],
        "pl_name": ["Venus", "Earth", "Mars"],
        "Ro": [ro_sol, ro_sol, ro_sol],
        "dRo": [dro_sol, dro_sol, dro_sol],
        "MHC": [r_p_venus / ra_sol, r_p_earth / ra_sol, r_p_mars / ra_sol],
        "dMHC": [dra_sol, dra_sol, dra_sol],
        "mass_class": [1, 1, 0],
        "habitable": [1, 1, 1]
    }

    if use_names:
        sol_data.update({"plot_name": ["Venus", "Earth", "Mars"]})
        df["plot_name"] = df["plot_name"].apply(lambda x: str(int(x)) if isinstance(x, (float, int)) else x)

    sol = pd.DataFrame(sol_data)

    return pd.concat([df, sol], ignore_index=True)


if __name__ == "__main__":

    # xlim = (0.0, 2.0)
    # ylim = (0.0, None)
    group = "mass_class"
    add_sol = True
    names_st = False

    xlim_a = (0.0, 5.0)
    ylim_a = (0.06, 60.0)
    # ylim_a = None
    xlim_h = (0.0, 4.2)
    ylim_h = (0.2, 30.0)
    # ylim_h = None
    xlim_g = (0.0, 4.0)
    # xlim_g = None
    ylim_g = (0.09, 1.0)
    # ylim_g = None
    xlim_h2 = (0.0, 4.2)
    xlim_h2 = None
    ylim_h2 = (27.0, 33.0)
    # ylim_h2 = None

    # Load planet habitability and plotting data
    df = pd.read_csv('current-exo-data/alfven_data.csv')
    df_h = df[df["habitable"] == 1].reset_index()
    df_h2 = df[(df["habitable"]) & (df["MHC"] > 1) & ((df["mass_class"] == 1) | (df["mass_class"] == 2))]
    df_g = df[(df["mass_class"] == 3) & (df["MHC"] < 1)].reset_index()
    df_g_dist = df_g[df_g["sy_dist"].notnull()]

    named_stars = df_h[(df_h[group] == 1) & (df_h["MHC"] > 1)]["hostname"].drop_duplicates().to_list()
    named_planets = df_h[(df_h[group] == 1) & (df_h["MHC"] > 1)]["pl_name"].to_list()
    named_objs = named_stars if names_st else named_planets

    plot_kwargs = {"alpha": 1.0, "s": 5, "s_h": 25, "lw_e": 1.0, "alpha_e": 0.25}

    # # master plots
    save_path_m1 = "imgs/Fig1.png"
    plot_proc(df, "Ro", "MHC", group=group, save_path=save_path_m1, xlim=xlim_a,
        ylim=ylim_a, named_objs=None, logy=True, plot_err=False,
        names_table=False, color_list=COLORS_1, reverse_grp=True,
        highlight_habitable=True, include_sol=True, **plot_kwargs)

    save_path_m2 = "imgs/Fig2.png"
    plot_err = [False, True, False, False]
    plot_proc(df_h, "Ro", "MHC", group=group, save_path=save_path_m2, xlim=xlim_h,
        ylim=ylim_h, named_objs=named_objs, logy=True, plot_err=plot_err,
        names_table=False, names_st=names_st, color_list=COLORS_1,
        reverse_grp=True, highlight_habitable=False, xnudge=0.01, ynudge=0.01,
        include_sol=True, sort_col="MHC", **plot_kwargs)

    save_path_e1 = "imgs/extra1.png"
    plot_proc(df_g, "Ro", "MHC", save_path=save_path_e1, xlim=xlim_g, ylim=ylim_g,
              logy=True, color_list=COLORS_1, **plot_kwargs)
    
    save_path_e2 = "imgs/extra2.png"
    plot_proc(df_h2, "Ro", "LX", group=group, save_path=save_path_e2,
              xlim=xlim_h2, ylim=ylim_h2, colors=COLORS_1, ylabel=r"$L_X$",
              **plot_kwargs)
    
    save_path_e2e = "imgs/extra2_err.png"
    plot_proc(df_h2, "Ro", "LX", group=group, save_path=save_path_e2e,
              xlim=xlim_h2, ylim=ylim_h2, colors=COLORS_1, errycol="dLX",
              ylabel=r"$L_X$", plot_err=plot_err, **plot_kwargs)

    save_path_e3 = "imgs/extra3.png"
    plot_proc(df_g_dist, "sy_dist", "MHC", save_path=save_path_e3,
              logy=False, color_list=COLORS_1, xlabel="Distance to Planetary System (pc)",
              ylim=(0.0, 1.0), **plot_kwargs)
    
    save_path_e3e = "imgs/extra3_err.png"
    plot_proc(df_g_dist, "sy_dist", "MHC", errxcol="sy_disterr", save_path=save_path_e3e,
              plot_err=True, logy=False, color_list=COLORS_1, xlabel="Distance to Planetary System (pc)",
              ylim=(0.0, 1.0), **plot_kwargs)
