from astropy.io import ascii
from astropy.table import Table
from astropy import units as u
import numpy as np
import os
import pandas as pd


table_cols = [
    "pl_name", "hostname",
    "sy_vmag", "e_sy_vmag",
    "sy_kmag", "e_sy_kmag",
    "st_mass", "e_st_mass",
    "Tauc", "e_Tauc",
    "Prot", "e_Prot", "r_Prot",
    "Ro", "e_Ro",
    "RA", "e_RA",
    "RASun", "e_RASun",
    "pl_orbsmax", "e_pl_orbsmax",
    "pl_orbeccen", "e_pl_orbeccen",
    "MHC", "e_MHC",
    "pl_bmasse", "e_pl_bmasse",
    "pl_rade", "e_pl_rade",
    "st_age", "e_st_age",
]

descriptions = {
        "pl_name": "Planet Name",
        "hostname": "Host Star Name",
        "Prot": "Stellar Rotation Period",
        "e_Prot": "Uncertainty in Prot",
        "r_Prot": "Source of Prot",
        "Tauc": "Convective Turnover Time",
        "e_Tauc": "Derived Uncertainty in Tauc",
        "Ro": "Rossby Number",
        "e_Ro": "Derived Uncertainty in Ro",
        "RA": "Host Star's Alfven Radius (au)",
        "e_RA": "Uncertainty in Alfven Radius (au)",
        "RASun": "Host Star's Alfven Radius (Rsun)",
        "e_RASun": "Uncertainty in Alfven Radius (Rsun)",
        "MHC": "Mag. Habitability Criterion",
        "e_MHC": "Derived Uncertainty in MHC",
        "mass_class": "Planet Mass Class",
        "rad_class": "Planet Radius Class",
        "size_class": "Planet Size Class",
        "st_mass": "Stellar Mass",
        "e_st_mass": "Uncertainty in Stellar Mass",
        "rperi": "Planetary Orbit Periastron",
        "e_rperi": "Uncertainty in Periastron",
        "sy_vmag": "V magnitude of star",
        "e_sy_vmag": "Uncertainty in V mag",
        "sy_kmag": "Ks magnitude of star",
        "e_sy_kmag": "Uncertainty in Ks mag",
        "pl_orbeccen": "Planet Orbital Eccentricity",
        "e_pl_orbeccen": "Uncertainty in Eccentricity",
        "pl_orbsmax": "Planet Orbital Semi-major Axis",
        "e_pl_orbsmax": "Uncertainty in Semi-major Axis",
        "VK_color": "V - Ks Color of Host Star",
        "e_VK_color": "Derived Uncertainty in V - Ks Color",
        "n_size_class": "Planet Size Class Subscript",
        "pl_bmasse": "Planet Mass",
        "e_pl_bmasse": "Uncertainty in Planet Mass",
        "pl_rade": "Planet Radius",
        "e_pl_rade": "Uncertainty in Planet Radius",
        "st_age": "Stellar Age",
        "e_st_age": "Uncertainty in Stellar Age",
}

u_mag = u.Unit("mag")
u_day = u.Unit("d")
u_Msun = u.Unit("Msun", format="cds")
u_Rsun = u.Unit("Rsun", format="cds")
u_au = u.Unit("au")
u_Mgeo = u.Unit("Mgeo", format="cds")
u_Rgeo = u.Unit("Rgeo", format="cds")
u_Gyr = u.Unit("Gyr")
u_dimless = u.Unit()

u_Mgeo._names.reverse()
u_Mgeo._short_names = ["Mgeo"]
u_Mgeo._long_names = ["geoMass"]

units = {
    "sy_vmag": u_mag,
    "e_sy_vmag": u_mag,
    "sy_kmag": u_mag,
    "e_sy_kmag": u_mag,
    "st_mass": u_Msun,
    "e_st_mass": u_Msun,
    "Prot": u_day,
    "e_Prot": u_day,
    "Tauc": u_day,
    "e_Tauc": u_day,
    "RA": u_au,
    "e_RA": u_au,
    "RASun": u_Rsun,
    "e_RASun": u_Rsun,
    "pl_orbsmax": u_au,
    "e_pl_orbsmax": u_au,
    "pl_bmasse": u_Mgeo,
    "e_pl_bmasse": u_Mgeo,
    "pl_rade": u_Rgeo,
    "e_pl_rade": u_Rgeo,
    "st_age": u_Gyr,
    "e_st_age": u_Gyr
}

latexnames = {
    "pl_name": "Planet Name",
    "hostname": "Host Name",
    "sy_vmag": "V",
    "e_sy_vmag": r"$\delta$ V",
    "sy_kmag": r"$K_s$",
    "e_sy_kmag": r"$\delta K_s$",
    "st_mass": r"$M_*$",
    "e_st_mass": r"$\delta M_*$",
    "Prot": r"$P_{rot}$",
    "e_Prot": r"$\delta P_{rot}$",
    "r_Prot": r"$P_{rot}$ Ref.",
    "Tauc": r"$\tau_c$",
    "e_Tauc": r"$\delta\tau_c$",
    "Ro": "Ro",
    "e_Ro": r"$\delta$ Ro",
    "RA": r"$R_A$",
    "e_RA": r"$\delta R_A$",
    "RASun": r"$R_A$",
    "e_RASun": r"$\delta R_A$",
    "pl_orbsmax": "a",
    "e_pl_orbsmax": r"$\delta$ a",
    "pl_orbeccen": "e",
    "e_pl_orbeccen": r"$\delta$ e",
    "MHC": "MHC",
    "e_MHC": r"$\delta$ MHC",
    "pl_bmasse": r"$M_{pl}$",
    "e_pl_bmasse": r"$\delta M_{pl}$",
    "pl_rade": r"$R_{pl}$",
    "e_pl_rade": r"$\delta R_{pl}$",
    "st_age": "Host Age",
    "e_st_age": r"$\delta$ Host Age"
}

latexcols = [
    "pl_name",
    "sy_vmag",
    "sy_kmag",
    "st_mass",
    "Prot",
    "Tauc",
    "Ro",
    "RASun",
    "pl_orbsmax",
    "pl_orbeccen",
    "MHC",
    "pl_bmasse",
    "pl_rade",
    "st_age",
]

def round_columns(table: pd.DataFrame) -> None:

    def round_col(col, dig, err: bool=True):
        table[col] = table[col].round(dig)
        if err:
            col = "e_" + col
            table[col] = table[col].round(dig)

    round_col("sy_vmag", 3)
    round_col("sy_kmag", 3)
    round_col("VK_color", 3)
    round_col("st_mass", 2)
    round_col("Prot", 3)
    round_col("Tauc", 3)
    round_col("Ro", 3)
    round_col("RA", 3)
    round_col("RASun", 3)
    round_col("pl_orbsmax", 3)
    round_col("pl_orbeccen", 3)
    round_col("MHC", 3)
    round_col("pl_bmasse", 2)
    round_col("pl_rade", 2)
    round_col("st_age", 2)


def make_mrt(table_mrt: Table):

    # descriptions
    for col in table_cols:
        desc = descriptions.get(col, "")
        table_mrt[col].description = desc

    tempfname = "temp_tab1.txt"
    ascii.write(table_mrt, tempfname, overwrite=True, format="mrt")

    # Meta-table header
    meta = "Title: Exploring the Magnetic Habitability of Exoplanets\n" + \
        "Authors: Anthony Atkinson, David Alexander, and Alison Farrish\n" + \
        "Table: Properties of Planets in our Magnetic Habitability Criterion Sample\n" + \
        "================================================================================\n" + \
        "Byte-by-byte Description of file: tab1.txt\n"

    f = open(tempfname, "r+")
    f.seek(0,0)
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    f_table = f.read(os.stat(tempfname).st_size)
    f.close()

    mrtfname = "tab1.txt"
    f_mrt = open(mrtfname, "w")
    f_mrt.write(meta)
    f_mrt.write(f_table)
    f_mrt.close()


def make_tex():

    mrtfname = "tab1.txt"
    tab = ascii.read(mrtfname, format="mrt").to_pandas()
    tab1: pd.DataFrame = tab.loc[:30, latexcols]
    tab1.rename(columns=latexnames, inplace=True)

    newunits = {latexnames[k]: units.get(k) for k in latexcols}
    tab1 = Table.from_pandas(tab1, units=newunits)

    caption = r"Properties of Planets in our Magnetic Habitability Criterion Sample"
    col_align = "l" + "c" * (len(latexcols) - 1)
    comment = r"\tablecomments{Table 1 is published in its entirety in the machine-readable format. " + \
                "A portion is shown here for guidance regarding its form and content." + \
                "The complete version includes uncertainties on measured and derived quantities, " + \
                "Alfv\'en Radius in AU, host star names, and references for rotation period measurements.}"
    preamble = "\n".join([r"\rotate",
                          r"\tablenum{1}",
                          r"\tablecolumns{14}",
                          r"\tabletypesize{\footnotesize}",
                          r"\tablewidth{0pt}"])

    latexdict = {
        "caption": caption,
        "col_align": col_align,
        "preamble": preamble,
        "tablefoot": comment,
        "tabletype": "deluxetable*",
        "units": units
    }

    texfname = "tab1.tex"
    ascii.write(table=tab1, output=texfname, Writer=ascii.AASTex, latexdict=latexdict, overwrite=True)


    cmd = r'sed -E -i -e "s/\&\s*$/\& \\\\\\\\/" -e "s/\&\s+(\&|\\\\\\\\)/\& \\\\nodata \1/g" tab1.tex'
    os.system(cmd)


def make_tex2(df: pd.DataFrame):

    tab2cols = ["pl_name",
                "VK_color", "e_VK_color",
                "st_mass", "e_st_mass",
                "Ro", "e_Ro",
                "MHC", "e_MHC",
                "mass_class", "rad_class"]

    tab2units = {
        r"$V - K_s$": r"$\Delta mag$",
        r"$M_*$": "Msun"
    }

    tab2names = {
        "pl_name": "Planet Name",
        "vk_str": r"$V - K_s$",
        "stm_str": r"$M_*$",
        "ro_str": "Stellar Rossby Number",
        "mhc_str": "MHC"
    }

    tab2tnames = {
        "NoFig2": "No. on Fig. 2",
        **tab2names
    }

    tab2tlatexdict = {
        "caption": r"Properties of Terran CHZ Planets with MHC $> 1$",
        "col_align": "lccccc",
        "preamble": r"\tablenum{2}",
        "tablefoot": r"\tablecomments{" + \
             "These 11 Terran planets (Earth-sized) are classified as such based on mass measurements fitting within " + \
             r"the range $0.5-3.0M_\Earth$.}",
        "tabletype": "deluxetable*"
    }

    tab2stlatexdict = {
        "caption": r"Properties of Superterran CHZ Planets with MHC $> 1$",
        "col_align": "lcccc",
        "preamble": r"\tablenum{3}",
        "tablefoot": r"\tablecomments{" + \
             "These 20 Superterran planets (Super-Earth-sized) are classified as such based on mass measurements fitting within " + \
             r"the range $3.0-10.0M_\Earth$.}",
        "tabletype": "deluxetable*"
    }
    
    df_h = df.loc[(df["habitable"] == 1) & (df["MHC"] > 1), tab2cols]
    terrans = df_h.loc[df_h["mass_class"] == 1].sort_values(by="MHC")
    superterrans = df_h.loc[df_h["mass_class"] == 2].sort_values(by="MHC")

    def fill(row, col: str):
        s = rf"${row[col]}"
        unc_col = f"e_{col}"
        if not np.isnan(row[unc_col]):
            s += rf"\pm{row[unc_col]}"
        
        s += "$"
        return s


    # terran
    terrans["vk_str"] = terrans.apply(lambda r: fill(r, "VK_color"), axis=1)
    terrans["stm_str"] = terrans.apply(lambda r: fill(r, "st_mass"), axis=1)
    terrans["ro_str"] = terrans.apply(lambda r: fill(r, "Ro"), axis=1)
    terrans["mhc_str"] = terrans.apply(lambda r: fill(r, "MHC"), axis=1)
    terrans["NoFig2"] = terrans.groupby("MHC", sort=False).ngroup() + 1
    tab2tcols = ["pl_name", "vk_str", "stm_str", "ro_str", "mhc_str", "NoFig2"]
    terrans_tex = terrans[tab2tcols]
    terrans_tex.rename(columns=tab2tnames, inplace=True)

    # superterran
    superterrans["vk_str"] = superterrans.apply(lambda r: fill(r, "VK_color"), axis=1)
    superterrans["stm_str"] = superterrans.apply(lambda r: fill(r, "st_mass"), axis=1)
    superterrans["ro_str"] = superterrans.apply(lambda r: fill(r, "Ro"), axis=1)
    superterrans["mhc_str"] = superterrans.apply(lambda r: fill(r, "MHC"), axis=1)
    tab2stcols = ["pl_name", "vk_str", "stm_str", "ro_str", "mhc_str"]
    superterrans_tex = superterrans[tab2stcols]
    superterrans_tex.rename(columns=tab2names, inplace=True)



    table_t = Table.from_pandas(terrans_tex, units=tab2units)
    table_st = Table.from_pandas(superterrans_tex, units=tab2units)
    
    tfname = "tab2t.tex"
    stfname = "tab2st.tex"
    ascii.write(table=table_t, output=tfname, Writer=ascii.AASTex, latexdict=tab2tlatexdict, overwrite=True)
    ascii.write(table=table_st, output=stfname, Writer=ascii.AASTex, latexdict=tab2stlatexdict, overwrite=True)

    cmd = r'sed -E -i -e "s/\&\s*$/\& \\\\\\\\/" -e "s/\&\s+(\&|\\\\\\\\)/\& \\\\nodata \1/g" tab2t.tex'
    os.system(cmd)

    cmd = r'sed -E -i -e "s/\&\s*$/\& \\\\\\\\/" -e "s/\&\s+(\&|\\\\\\\\)/\& \\\\nodata \1/g" tab2st.tex'
    os.system(cmd)

    cmd = r'sed -E -i -e "s/\&\s+\&/\&/g" tab2t.tex tab2st.tex'
    os.system(cmd)


if __name__ == "__main__":

    do_mrt = True
    do_tex = True
    do_tex2 = True

    fname = "current-exo-data/alfven_data.csv"
    table = ascii.read(fname, format="csv").to_pandas()
    round_columns(table)
    table.sort_values(by="pl_name", inplace=True)
    table_mrt = Table.from_pandas(table.loc[:,table_cols], units=units)

    if do_mrt:
        make_mrt(table_mrt)
    if do_tex:
        make_tex()
    if do_tex2:
        make_tex2(table)
