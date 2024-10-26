from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import colormaps as cm
import numpy as np
import pandas as pd
import re

CMAP3 = ListedColormap([
    "firebrick",
    "orange",
    "yellow",
    "lightyellow",
    # "lightcyan",
    # "paleturquoise",
    # "turquoise"
], "CMAP3")

cm.register(cmap=CMAP3)

SPECTYPE_STRS = [
    "M",
    "K",
    "G",
    "F",
    # "A",
    # "B",
    # "O"
]
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


def _measured_uncertainties(nasa_exo: pd.DataFrame) -> pd.DataFrame:
    
    df = nasa_exo
    err_calc = lambda e1, e2: np.max([e1, np.fabs(e2)], axis=0)

    params = {}
    p = re.compile("(.+)err(1|2)$")
    for col in nasa_exo.columns:
        # find parameter name that ends with err1 or err2 and extract the prefix
        match = p.match(col)
        if match is None:
            continue
        param = match.group(1)
        params[param] = params.get(param, 0) + 1
        
        # Only calculate error if only two error parameters are available in table
        if params[param] == 2:
            err1 = df[param + "err1"]
            err2 = df[param + "err2"]
            df["e_" + param] = err_calc(err1, err2)
    
    return nasa_exo


def _underscore_to_dash(df: pd.DataFrame):

    params = {}
    p = re.compile("^(?!e_).+")
    for col in df.columns:
        match = p.match(col)
        if match is None:
            continue
        s = match.group()
        params[s] = s.replace("_", "-")

    return df.rename(columns=params)


def plot1(df: pd.DataFrame):
    
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax: plt.Axes


    df_grp = df.groupby("st-spectype-col")

    x_grp = []
    labels = []
    for g in df_grp.groups.keys():
        grp = df_grp.get_group(g)
        x_grp.append(np.log10(grp["pl-orbsmax"]))
        labels.append(f"{SPECTYPE_STRS[int(g)]} ({grp["pl-orbsmax"].count()})")

    # x_grp = [np.log10(df_grp.get_group(g)["pl-orbsmax"]) for g in df_grp.groups.keys()]
    # labels = [f"{st}" for st in SPECTYPE_STRS]



    # x = np.log10(df["pl-orbsmax"])
    


    # z = df["st-spectype-col"].astype("Int64")
    # cmap = cm.get_cmap("CMAP3")

    # ax.tick_params(labelsize=14, size=5)
    # im = ax.scatter(x, y, c=z, cmap=cmap, marker="o",edgecolors="black", s=s, zorder=2)
    # cb = fig.colorbar(im, ax=ax, aspect=13)
    # cb_ax = cb.ax
    # cb_ax.yaxis.set_ticks([])
    # for i, st in enumerate(SPECTYPE_STRS):
    #     ct = np.sum(np.all(np.bitwise_not(np.isnan(np.stack((x, y, s, z)))), axis=0) & (df["st-spectype-ltr"] == st))
    #     cb_ax.text(0.5, (i + 0.5) * (len(SPECTYPE_STRS) - 1) / len(SPECTYPE_STRS), f"{st}\n({ct})", ha="center", va="center", fontsize=12)
    # cb_ax.yaxis.labelpad = 25
    # cb_ax.set_ylabel("Spectral Type", rotation=270, fontsize=18)
    
    ax.hist(x_grp, bins=20, histtype="barstacked", edgecolor="black", color=CMAP3.colors, label=labels)

    ax.set_xlabel(rf"$\log_{{10}}$ a (au)", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)
    ax.set_title("Hot Jupiter distribution in orbital semi-major axis", fontsize=18)
    ax.tick_params(labelsize=14, size=10)
    ax.legend(loc=1)
    plt.show()


if __name__ == "__main__":

    use_NEA: bool=False

    if use_NEA:
        df = pd.read_csv("tables-merged/nasa_exo.csv")
        df = _measured_uncertainties(df)
        df = _underscore_to_dash(df)
    else:
        df = pd.read_csv('tables-merged/alfven_data.csv')

    df_sp = _parse_spectype(df["st-spectype"])
    df = df.merge(df_sp, how="outer", left_index=True, right_index=True, suffixes=(None, "_1"))

    # Load plotting data for HJs    
    criteria = (df["pl-bmasse"] / 317.8 > 0.25) & \
               (df["pl-bmasse"] / 317.8 < 13)
    df_hj = df[criteria].reset_index()

    plot1(df_hj)