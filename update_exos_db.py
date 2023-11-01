import numpy as np
import pandas as pd
import re
from os import system

# FETCH NEW EXO DATA:
# > cat nasa_exo_query.txt | xargs wget -o nasa_exo_PSCP.csv
fetch_DBs = True   # Last updated: 2023.10.20:15:48 CDT
update_dbs = True
merge_dbs = True

'''
DATA/COLUMNS THAT ARE NEEDED
 - Prot/PRot/st_rotp
 - pl_name (Planet Name)
   - pl_letter
   - kic_id/kepid
   - koi/koi_name
   - tic_id
   - toi
   - gaia_id
 - pl_bmasse (Planet Mass in Earth Masses)
 - pl_rade (Planet Radius in Earth Radii)
 - pl_orbsmax (Planet Orbit Semi-Major Axis in AU)
 - pl_orbeccen (Planet Orbit Eccentricity)

 - sy_vmag (V-band magnitude of star)
 - sy_kmag (K-band magnitude of star)
 - st_rad (Stellar Radius in Solar Radii)
 - st_mass (Stellar Mass in Solar Masses)

 - notes:
   - ALL parameters EXCEPT names/ids and Prots are available via nasa/caltech
     - currently selecting names+prots from surveys and merging onto nasa
   - may want to use consistent parameter set (dont use PSCP db, instead use
   PS database and select only where 'Default Parameter Set is 1') or parameter
   set offered by the paper offering the Prot
'''


def init_nasa_exo():

    # Load NASA exoplanet database
    nasa_exo = pd.read_csv("tables/nasa_exo_PSCP.csv")

    # !! need to fetch new caltech DB of Kep confs and update
    # Add KOI/KIC ID columns of confirmed Kepler planets
    p_koi = re.compile(r"^K(\d+)\.\d{2}$")
    nasa_kep = pd.read_csv("tables/nasa_exo_kep.csv")
    nasa_kep["KOI"] = nasa_kep.apply(lambda r: int(r.koi_name[1:-3]) if p_koi.match(str(r.koi_name)) is not None else np.nan, axis=1)
    nasa_kep.rename(columns={"kepid": "KIC"}, inplace=True)
    nasa_exo = pd.merge(nasa_exo, nasa_kep, how="left", on="pl_name")
    nasa_exo["KOI"].fillna(-1, inplace=True, downcast="infer")
    nasa_exo["KIC"].fillna(-1, inplace=True, downcast="infer")

    # Add TIC ID column
    p_tic = re.compile(r"^TIC (\d+)$")
    nasa_exo["TIC"] = nasa_exo.apply(lambda r: int(str(r.tic_id)[4:]) if p_tic.match(str(r.tic_id)) is not None else -1, axis=1)

    # Add GAIA ID columns
    nasa_exo["GAIA"] = nasa_exo.apply(lambda x: int(x["gaia_id"][9:]) if re.match(r"^Gaia DR2 \d+$", str(x["gaia_id"])) is not None else np.nan, axis=1)

    nasa_exo.to_csv("current-exo-data/nasa_exo.csv", index=False)
    return nasa_exo


def update_exos_sim10(nasa_exo: pd.DataFrame):

    # Load Simpson 2010 Rotation Period data
    sim10_prot = pd.read_csv("tables/custom_prot.txt", sep="\s+", header=1, nrows=11)

    sim10_prot = sim10_prot[["name","prot","eprot", "grade"]]
    sim10_prot.rename(columns={"name": "hd_name", "prot":"Prot","eprot":"e_Prot"}, inplace=True)
    sim10_prot.insert(len(sim10_prot.columns), "db", "sim10")
    sim10_prot = sim10_prot[sim10_prot["Prot"].notnull() & (sim10_prot["grade"] == "C")]
    # C: confirmed; P: probable; W: weak

    exos_sim10 = pd.merge(nasa_exo, sim10_prot, how="inner", on="hd_name")[["hostname","Prot","e_Prot", "db"]]
    exos_sim10.drop_duplicates(subset="hostname", inplace=True)
    exos_sim10.to_csv("current-exo-data/hosts_sim10.csv", index=False)
    return exos_sim10


# Update the short-list of exoplanet systems/stars shared by mcq13 and nasa
def update_exos_mcq13(nasa_exo: pd.DataFrame):

    # Load McQuillan 2013 KOI stellar rotation rate survey
    mcq13_kois = pd.read_csv("tables/prot_mcq_2013.dat", sep="\s+")
    mcq13_kois = mcq13_kois[["KOI","PRot", "e_PRot"]]
    mcq13_kois.rename(columns={"PRot":"Prot","e_PRot": "e_Prot"}, inplace=True)
    mcq13_kois.insert(len(mcq13_kois.columns), "db", "mcq13")
    mcq13_kois = mcq13_kois[mcq13_kois.Prot.notnull()]

    # Select stars from mcq13 whose KOI is in nasa db
    exos_mcq13 = pd.merge(nasa_exo, mcq13_kois, how="inner", on="KOI")[["hostname","KOI","Prot","e_Prot", "db"]]
    exos_mcq13.drop_duplicates(subset="hostname", inplace=True)
    exos_mcq13.to_csv("current-exo-data/hosts_mcq13.csv", index=False)
    return exos_mcq13


def update_exos_mcq14(nasa_exo: pd.DataFrame):

    # Load McQuillan 2014 KIC stellar rotation rate survey
    mcq14_kics = pd.read_csv("tables/prot_mcq_2014.dat", sep="\s+")
    mcq14_kics = mcq14_kics[["KIC","PRot", "e_PRot"]]
    mcq14_kics.rename(columns={"PRot":"Prot","e_PRot": "e_Prot"}, inplace=True)
    mcq14_kics.insert(len(mcq14_kics.columns), "db", "mcq14")
    mcq14_kics = mcq14_kics[mcq14_kics.Prot.notnull()]

    exos_mcq14 = pd.merge(nasa_exo, mcq14_kics, how="inner", on="KIC")[["hostname","KIC","Prot","e_Prot", "db"]]
    exos_mcq14.drop_duplicates(subset="hostname", inplace=True)
    exos_mcq14.to_csv("current-exo-data/hosts_mcq14.csv", index=False)
    return exos_mcq14


def update_exos_arm15(nasa_exo: pd.DataFrame):

    # Load Armstring 2015 Kepler flare, rotation, and activity of habitable pl survey
    arm15_prot = pd.read_csv("tables/custom_prot.txt", sep="\s+", header=14, nrows=7)
    
    arm15_prot = arm15_prot[["hostname","prot_acf","eprot_acf"]]
    arm15_prot.rename(columns={"prot_acf":"Prot","eprot_acf":"e_Prot"}, inplace=True)
    arm15_prot.insert(len(arm15_prot.columns), "db", "arm15")
    arm15_prot = arm15_prot[arm15_prot["Prot"].notnull()]

    exos_arm15 = pd.merge(nasa_exo, arm15_prot, how="inner", on="hostname")[["hostname","Prot","e_Prot", "db"]]
    exos_arm15.drop_duplicates(subset="hostname", inplace=True)
    exos_arm15.to_csv("current-exo-data/hosts_arm15.csv", index=False)
    return exos_arm15


def update_exos_mar20(nasa_exo: pd.DataFrame):

    # Load Martin 2020 TOI stellar rotation period survey
    mar20 = pd.read_csv("tables/prot_martin_2020_tic.csv", usecols=[0,1,2,3,4])
    mar20["TIC"] = mar20.apply(lambda x: int(x["TIC_ID"]), axis=1)
    mar20.drop(columns="TIC_ID",inplace=True)
    mar20.rename(columns={"eProt":"e_Prot"}, inplace=True)
    mar20.insert(len(mar20.columns), "db", "mar20")
    mar20 = mar20[mar20.Prot.notnull()]

    # Selection criteria for rotation rates based on their classification/certainty
    selection = (mar20["Variability_Classification"] == "Unambiguous_Rotation") & mar20["Prot"].notnull()
    # selection = ((mar20["Variability_Classification"] == "Unambiguous_Rotation") | (mar20["Variability_Classification"] == "Dubious_Rotation")) & mar20["Prot"].notnull()
    mar20_prot = mar20[selection].copy(deep=False)
    mar20_prot["Prot"] = mar20_prot.apply(lambda x: select_prot_martin(x["Prot"]), axis=1)

    exos_mar20 = pd.merge(nasa_exo, mar20_prot, how="inner", on="TIC")[["hostname","TIC","Prot","e_Prot", "db"]]
    exos_mar20.drop_duplicates(subset="hostname", inplace=True)
    exos_mar20.to_csv("current-exo-data/hosts_mar20.csv", index=False)
    return exos_mar20


# fn can be np.mean/min/max, lambda x: np.nan
def select_prot_martin(prot_str: str, fn=np.min):
    if "/" in str(prot_str):
        prots = prot_str.split("/")
        protf = []
        for s in prots:
            protf.append(float(s))
        return fn(protf)
    else: return float(prot_str)


def update_exos_lu22(nasa_exo: pd.DataFrame):

    # Load Lu 2022 ZTF rotation period survey
    lu22_gaia = pd.read_csv("tables/prot_lu_gaia.txt", header=21, sep="\s+")

    lu22_gaia = lu22_gaia[["GAIA","Prot"]]
    lu22_gaia.insert(len(lu22_gaia.columns), "db", "lu22")
    lu22_gaia = lu22_gaia[lu22_gaia.Prot.notnull()]

    # problem here: discrepancy in number of entries
    # exos_lu22 = lu22_gaia[lu22_gaia.GAIA.isin(nasa_exo.GAIA)]

    exos_lu22 = pd.merge(nasa_exo, lu22_gaia, how="inner", on="GAIA")[["hostname","GAIA","Prot", "db"]]
    exos_lu22.drop_duplicates(subset="hostname", inplace=True)

    exos_lu22.to_csv("current-exo-data/hosts_lu22.csv",index=False)
    return exos_lu22


def update_exos_nasa(nasa_exo: pd.DataFrame):

    # Load NASA Exoplanet Archive rotation period entries
    exos_nasa = nasa_exo[nasa_exo.st_rotp.notnull()][["hostname", "st_rotp", "st_rotperr1", "st_rotperr2"]]
    exos_nasa["st_rotperr"] = nasa_exo.apply(lambda r: np.max([r.st_rotperr1, np.fabs(r.st_rotperr2)]), axis=1)
    exos_nasa.rename(columns={"st_rotp": "Prot", "st_rotperr": "e_Prot"}, inplace=True)
    exos_nasa.drop_duplicates(subset="hostname", inplace=True)
    exos_nasa.drop(columns=["st_rotperr1", "st_rotperr2"], inplace=True)
    exos_nasa.insert(len(exos_nasa.columns), "db", "nasa")

    exos_nasa.to_csv("current-exo-data/hosts_nasa.csv", index=False)
    return exos_nasa


def update_exos_habitable(nasa_exo: pd.DataFrame):
    habitable = pd.read_csv("tables/habitable.txt", header=1)
    exos_habitable = pd.merge(nasa_exo, habitable, how="inner", on="pl_name")[["hostname", "pl_name"]]
    exos_habitable.to_csv("current-exo-data/exos_habitable.csv", index=False)


def update_exos_hill23(nasa_exo: pd.DataFrame):
    # decide what to do w all the new measurements
    # check Hill - if it just uses NEA then it shouldn't mater
    # i guess we should also compare via computer too
    hill23 = pd.read_csv("tables/CHZ_hill23.csv")
    hill23 = hill23[["Planet"]]
    hill23.rename(columns={"Planet": "pl_name"}, inplace=True)
    exos_hill23 = pd.merge(nasa_exo, hill23, how="inner", on="pl_name")[["hostname", "pl_name"]]
    exos_hill23.to_csv("current-exo-data/exos_hill23.csv", index=False)


if __name__ == "__main__":

    if fetch_DBs:
        print("Updating Planetary Systems Composite Parameters (pscomppars) database!")
        system("cat nasa_exo_query.txt | xargs wget -o log_PSCP -O tables/nasa_exo_PSCP.csv")

        print("Updating Kepler confirmed planets (kepnames) database!")
        system("cat nasa_exo_kep_query.txt | xargs wget -o log_kep -O tables/nasa_exo_kep.csv")

    if update_dbs:
        nasa_exo = init_nasa_exo()
        hosts_sim10 = update_exos_sim10(nasa_exo)
        hosts_mcq13 = update_exos_mcq13(nasa_exo)
        hosts_mcq14 = update_exos_mcq14(nasa_exo)
        hosts_arm15 = update_exos_arm15(nasa_exo)
        hosts_mar20 = update_exos_mar20(nasa_exo)
        hosts_lu22 = update_exos_lu22(nasa_exo)
        hosts_nasa = update_exos_nasa(nasa_exo)
        update_exos_habitable(nasa_exo)
        update_exos_hill23(nasa_exo)

        print("[nasa_exo]\nplanets {}\n".format(nasa_exo["pl_name"].count()))
        print("[sim10]\nProt {}\ne_Prot {}\n".format(
            hosts_sim10["Prot"].count(), hosts_sim10["e_Prot"].count()))
        print("[mcq13]\nProt {}\ne_Prot {}\n".format(
            hosts_mcq13["Prot"].count(), hosts_mcq13["e_Prot"].count()))
        print("[mcq14]\nProt {}\ne_Prot {}\n".format(
            hosts_mcq14["Prot"].count(), hosts_mcq14["e_Prot"].count()))
        print("[arm15]\nProt {}\ne_Prot {}\n".format(
            hosts_arm15["Prot"].count(), hosts_arm15["e_Prot"].count()))
        print("[mar20]\nProt {}\ne_Prot {}\n".format(
            hosts_mar20["Prot"].count(), hosts_mar20["e_Prot"].count()))
        print("[lu22]\nProt {}\n".format(hosts_lu22["Prot"].count()))
        print("[nasa]\nProt {}\ne_Prot {}\n".format(
            hosts_nasa["Prot"].count(), hosts_nasa["e_Prot"].count()))

    if merge_dbs:
        hosts_sim10 = pd.read_csv("current-exo-data/hosts_sim10.csv")
        hosts_mcq13 = pd.read_csv("current-exo-data/hosts_mcq13.csv")
        hosts_mcq14 = pd.read_csv("current-exo-data/hosts_mcq14.csv")
        hosts_arm15 = pd.read_csv("current-exo-data/hosts_arm15.csv")
        hosts_mar20 = pd.read_csv("current-exo-data/hosts_mar20.csv")
        hosts_lu22 = pd.read_csv("current-exo-data/hosts_lu22.csv")
        hosts_nasa = pd.read_csv("current-exo-data/hosts_nasa.csv")

        hosts = pd.concat([hosts_sim10, hosts_mcq13, hosts_mcq14, hosts_arm15, hosts_mar20, hosts_lu22, hosts_nasa], ignore_index=True)
        hosts["e_Prot"] = hosts.apply(lambda r: np.nan if r.e_Prot == 0 else r.e_Prot, axis=1)
        hosts["KOI"].fillna(-1, inplace=True, downcast="infer")
        hosts["KIC"].fillna(-1, inplace=True, downcast="infer")
        hosts["TIC"].fillna(-1, inplace=True, downcast="infer")
        hosts["GAIA"].fillna(-1, inplace=True, downcast="infer")

        hosts.sort_values("e_Prot", inplace=True, ignore_index=True, ascending=True)
        hosts.drop_duplicates(subset="hostname", inplace=True, keep="first")
        hosts = hosts[["hostname", "Prot", "e_Prot", "KOI", "KIC", "TIC", "GAIA", "db"]]
        hosts.to_csv("current-exo-data/hosts_prot.csv", index=False)
        
        print("[hosts]\nProt {}\ne_Prot {}\n".format(
            hosts["Prot"].count(), hosts["e_Prot"].count()))