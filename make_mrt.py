from astropy.io import ascii
from astropy.table import Table
import os
import pandas as pd


table_cols = [
    "pl_name", "hostname",
    "sy_vmag", "e_sy_vmag",
    "sy_kmag", "e_sy_kmag",
    "Tauc", "e_Tauc",
    "pl_orbsmax", "e_pl_orbsmax",
    "pl_orbeccen", "e_pl_orbeccen",
    "pl_bmasse", "e_pl_bmasse",
    "pl_rade", "e_pl_rade",
    "MHC", "e_MHC",
    "Ro", "e_Ro",
    "RA", "e_RA",
    "st_age", "e_st_age",
    "Prot", "e_Prot", "r_Prot"
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
        "RA": "Radius of Host Star's Mean Alfven Surface",
        "e_RA": "Uncertainty in Alfven Radius",
        "MHC": "Magnetic Habitability Criterion",
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

units = {
    "Prot": "d",
    "e_Prot": "d",
    "RA": "au",
    "e_RA": "au",
    "rperi": "au",
    "e_rperi": "au",
    "VK_color": "mag",
    "st_age": "Gyr",
    "e_st_age": "Gyr",
    "st_mass": "solMass",
    "e_st_mass": "solMass",
    "pl_bmasse": "Mgeo",
    "e_pl_bmasse": "Mgeo",
    "pl_rade": "Rgeo",
    "e_pl_rade": "Rgeo"
}


if __name__ == "__main__":

    fname = "current-exo-data/alfven_data.csv"
    table = ascii.read(fname, format="csv").to_pandas()
    table: pd.DataFrame
    table = table.round(4)
    table.sort_values(by="pl_name", inplace=True)

    table_a = table.loc[:,table_cols]
    # table_h = table.loc[(table["habitable"] == 1) & (table["MHC"] > 1) & ((table["mass_class"] == 1) | (table["rad_class"] == 1)), table_cols]
    table_mrt = Table.from_pandas(table_a)

    for col, desc in descriptions.items():
        if col in table_mrt.colnames:
            table_mrt[col].description = desc

    for col, unit in units.items():
        if col in table_mrt.colnames:
            table_mrt[col].unit = unit

    tempfname = "temp_tab1.txt"
    ascii.write(table_mrt, tempfname, overwrite=True, format="mrt")

    meta = "Title: Exploring the Magnetic Habitability of Terrestrial Exoplanets in Circum-\nstellar Habitable Zones\n" + \
        "Authors: Anthony Atkinson, David Alexander, and Alison Farrish\n" + \
        "Table: <Table Name>\n"


    f = open(tempfname, "r+")
    f.seek(0,0)
    print(f.readline())
    print(f.readline())
    print(f.readline())
    f_table = f.read(os.stat(tempfname).st_size)
    f.close()
    print(f_table)

    mrtfname = "tab1.txt"
    f_mrt = open(mrtfname, "w")
    f_mrt.write(meta)
    f_mrt.write(f_table)
    f_mrt.close()
