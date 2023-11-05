from astropy.io import ascii
from astropy.table import Table
from astropy import units as u
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

# u_mag = u.Unit("mag")
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
    "pl_name": "Name",
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


def make_mrt():

    fname = "current-exo-data/alfven_data.csv"
    table = ascii.read(fname, format="csv").to_pandas()
    round_columns(table)

    table.sort_values(by="pl_name", inplace=True)

    table_a = table.loc[:,table_cols]
    table_mrt = Table.from_pandas(table_a)

    for col, desc in descriptions.items():
        if col in table_mrt.colnames:
            table_mrt[col].description = desc

    for col in table_cols:
        unit = units.get(col, u_dimless)
        table_mrt[col].unit = unit

    print(table_mrt["pl_bmasse"].unit.__dict__)

    tempfname = "temp_tab1.txt"
    ascii.write(table_mrt, tempfname, overwrite=True, format="mrt")

    # ascii.mrt

    meta = "Title: Exploring the Magnetic Habitability of Terrestrial Exoplanets in Circum-\nstellar Habitable Zones\n" + \
        "Authors: Anthony Atkinson, David Alexander, and Alison Farrish\n" + \
        "Table: Properties of Planets with Calculable MHC\n"


    f = open(tempfname, "r+")
    f.seek(0,0)
    print(f.readline())
    print(f.readline())
    print(f.readline())
    f_table = f.read(os.stat(tempfname).st_size)
    f.close()
    # print(f_table)

    mrtfname = "tab1.txt"
    f_mrt = open(mrtfname, "w")
    f_mrt.write(meta)
    f_mrt.write(f_table)
    f_mrt.close()


def make_tex():
    tab1 = ascii.read("tab1.txt", format="mrt")[:30][latexcols]

    for name in latexcols:
        if name in latexnames.keys():
            # print(name)
            # print(tab1[name])
            # print(tab1[name].mask)
            # tab1[tab1[name].mask][name] = r"\nodata"
            newname = latexnames[name]
            tab1.rename_column(name, newname)

    caption = "Properties of Planets with Calculable MHC"
    comment = r"\tablecomments{Table 1 is published in its entirety in the machine-readable format. " + \
                "A portion is shown here for guidance regarding its form and content." + \
                "The complete version includes uncertainties on measured and derived quantities, " + \
                "Alfv\'en Radius in AU, host star names, and references for rotation period measurements.}"
    
    latexdict = {
        "caption": caption,
        "tablefoot": comment,
        "tabletype": "deluxetable*",
        "units": units
    }

    ascii.write(table=tab1, output="tab1.tex", Writer=ascii.AASTex, latexdict=latexdict, overwrite=True)


if __name__ == "__main__":

    do_mrt = True
    do_tex = True

    if do_mrt:
        make_mrt()
    if do_tex:
        make_tex()
