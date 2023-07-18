import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


# CLASS_COLORS = ['#DDBB44','#00AA00','#DD5500','#33CCCC','#000000']
CLASS_COLORS = ['C0','C1','C2','C3','C4']
CLASS_LABELS = ['subterran', 'terran', 'superterran', 'giant','no class']
NAMED_STARS = ["Sol", "BD-11 4672", "GJ 1132", "GJ 3293", "K2-3", "Kepler-186",
               "Kepler-419", "Kepler-436", "Kepler-438", "Kepler-62", "Kepler-705", "LHS 1140",
               "Proxima Cen", "Ross 128", "Teegarden's Star", "TOI-700",
               "TRAPPIST-1", "Wolf 1061"]
SOL_NAMES = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn",
             "Uranus", "Neptune"]

def plot_df(df, ax: plt.Axes, logy=False, show_names=False, plot_err=False,
            xnudge: float=0.05, ynudge: float=0.05, reverse_grp: bool=False,
            highlight_habitable: bool=False, color_h: str="C9",
            marker_h: str="*"):

    x = np.array(df.Ro)
    y = np.array(df.orbit2alfven)
    grp_num = df.mass_class.iat[0]
    color = CLASS_COLORS[grp_num]
    label = CLASS_LABELS[grp_num]
    
    if logy:
        ax.set_yscale("log")
    
    if plot_err:
        xerr = np.array(df.dRo)
        yerr = np.array(df.dorbit2alfven)
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, linestyle="None", ecolor="#333333", elinewidth=1.1, alpha=0.4, zorder=0.0)

    if show_names:
        xtext = x + xnudge
        ytext = (y * (1 + ynudge)) if logy else (y + ynudge)
        name = df.plot_name
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
        ax.scatter(x_h, y_h, color=color_h, marker=marker_h, zorder=9.5, s=100,label="CHZ")
        ax.scatter(x_nh, y_nh, color=color, label=label, zorder=scatter_zorder)
    else:
        ax.scatter(x, y, color=color, label=label, zorder=scatter_zorder)
    
    ax.set_xlabel("Rossby Number", fontsize=18)
    ax.set_ylabel("MHC", fontsize=18)
    ax.tick_params(labelsize=16)

def _make_name(row: pd.DataFrame) -> str:
    
    s = str(row["grp_num"])
    if row["grp_len"] == 1:
        return s
    
    s += row["pl_letter"]
    if row["pl_name"] not in SOL_NAMES:
        return s
    
    return "%s (%s)"%(row["pl_name"], s)

def make_plot_names(df: pd.DataFrame, return_mapping: bool=False) -> pd.DataFrame:
    hostnames = pd.Series(NAMED_STARS, name="hostname")
    df_fltr = pd.merge(df, hostnames, how="inner", on="hostname")[["hostname", "pl_name", "pl_letter"]]
    df_fltr["grp_num"] = df_fltr.groupby("hostname").ngroup() + 1
    df_map = df_fltr.drop_duplicates(subset="hostname")[["grp_num", "hostname"]]
    # name_mapping = [{entry["grp_num"]: entry["hostname"]} for entry in df_map.to_dict(orient="records")]
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

def plot_proc(df: pd.DataFrame, group: str=None, save_path: str=None,
              xlim: tuple[float]=None, ylim: tuple[float]=None,
              show_names: bool=False, names_table: bool=False,
              logy: bool=False, bottom_adj: float=0.125, left_adj: float=0.08,
              leg_loc: int=4, **kwargs) -> None:

    if show_names:
        if names_table:
            df["plot_name"], name_mapping = make_plot_names(df, return_mapping=True)
        else:
            df["plot_name"] = make_plot_names(df)

    if group is not None:
        df_plot = df.groupby(group, as_index=False)
    else:
        df_plot = df

    ax1: plt.Axes
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    
    x = np.array(df["Ro"])
    y = np.array(df["orbit2alfven"])

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
    df_plot.apply(plot_df, ax=ax1, show_names=show_names, logy=logy, **kwargs)
    
    hand, labl = ax1.get_legend_handles_labels()
    _, index_u = np.unique(labl, return_index=True)
    labl_u = np.array(labl)[index_u]
    hand_u = np.array(hand)[index_u]
    
    ax1.legend(hand_u, labl_u, loc=leg_loc)

    if names_table:
        name_mapping = sorted(name_mapping, key=lambda x: x[0])
        ax1.table(cellText=name_mapping, colWidths=[0.05, 0.15], loc="right")

    plt.tight_layout()
    plt.subplots_adjust(bottom=bottom_adj, left=left_adj)
    if save_path is not None:
        fig.savefig(save_path)
    
    plt.show()


def add_solar_system_planets(df: pd.DataFrame) -> pd.DataFrame:
    ro_sol = 1.85
    dro_sol = 0.26
    ra_sol = 0.1
    dra_sol = 0.02
    r_p_venus = 0.718
    r_p_earth = 1.0
    r_p_mars = 1.381
    
    sol = pd.DataFrame({
        "hostname": ["Sol", "Sol", "Sol"],
        "pl_name": ["Venus", "Earth", "Mars"],
        "pl_letter": ["c", "d", "e"],
        "Ro": [ro_sol, ro_sol, ro_sol],
        "dRo": [dro_sol, dro_sol, dro_sol],
        "orbit2alfven": [r_p_venus / ra_sol, r_p_earth / ra_sol, r_p_mars / ra_sol],
        "dorbit2alfven": [dra_sol, dra_sol, dra_sol],
        "mass_class": [1, 1, 0],
        "habitable": [1, 1, 1]
    })

    return pd.concat([df, sol], ignore_index=True)

if __name__ == "__main__":

    # xlim = (0.0, 2.0)
    # ylim = (0.0, None)
    group = "mass_class"
    add_sol = True

    xlim_a = (0.0, 5.0)
    ylim_a = (0.06, 60.0)
    xlim_h = (0.0, 4.2)
    ylim_h = (0.2, 30.0)

    # Load planet habitability and plotting data
    df = pd.read_csv('current-exo-data/alfven_data.csv')

    if add_sol:
        df = add_solar_system_planets(df)

    # plot_list = [1, 2, 3, 4, 5, 6]
    plot_list = [2, 6]

    if 1 in plot_list:
        # save_path_a = "imgs/C0_MHC_log.png"
        save_path_a = None
        plot_proc(df, group=group, save_path=save_path_a, xlim=xlim_a, ylim=ylim_a,
                logy=True, left_adj=0.095, leg_loc=1)
    
    if 2 in plot_list:
        save_path_a = "imgs/C0_MHC_log_hbt.png"
        # save_path_a = None
        plot_proc(df, group=group, save_path=save_path_a, xlim=xlim_a, ylim=ylim_a,
                logy=True, left_adj=0.095, leg_loc=1, highlight_habitable=True)

    if 3 in plot_list:
        # save_path_r = "imgs/C0_MHC_log_rev.png"
        save_path_r = None
        plot_proc(df, group=group, save_path=save_path_r, xlim=xlim_a, ylim=ylim_a,
                logy=True, left_adj=0.095, leg_loc=1, reverse_grp=True)
    
    if 4 in plot_list:
        # save_path_h = "imgs/C0_MHC_CHZ_log.png"
        save_path_h = None
        df_h = df[df.habitable == 1].reset_index()
        plot_proc(df_h, group=group, save_path=save_path_h, xlim=xlim_h,
                ylim=ylim_h, show_names=True, logy=True, plot_err=False)
    
    if 5 in plot_list:
        # save_path_h_e = "imgs/C0_MHC_CHZ_log_err.png"
        save_path_h_e = None
        df_h = df[df.habitable == 1].reset_index()
        plot_proc(df_h, group=group, save_path=save_path_h_e, xlim=xlim_h,
                ylim=ylim_h, show_names=True, logy=True, plot_err=True)
    

    if 6 in plot_list:
        save_path_h_e_t = "imgs/C0_MHC_CHZ_log_err_tbl.png"
        # save_path_h_e_t = None
        df_h = df[df.habitable == 1].reset_index()
        plot_proc(df_h, group=group, save_path=save_path_h_e_t, xlim=xlim_h,
                ylim=ylim_h, show_names=True, logy=True, plot_err=True,
                names_table=True)

    