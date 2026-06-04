import pandas as pd

if __name__ == "__main__":

    # identify stars hosting HJ
    db = pd.read_csv("tables-merged/nasa_exo.csv")
    hj = (db["pl_bmassj"] > 0.25) & \
            (db["pl_bmassj"] < 13.6) & \
            (db["pl_orbsmax"] < 0.1)
    hj_hosts = db.loc[hj]
    
    # identify stars hosting planets in their CHZ
    chz_upr = pd.read_csv("tables-merged/exos_habitable.csv")
    chz_hill = pd.read_csv("tables-merged/exos_hill23.csv")
    chz_hosts = pd.merge(chz_upr, chz_hill, how="outer")

    hj_chz_hosts = pd.merge(hj_hosts, chz_hosts, how="inner", on="hostname",
                            suffixes=["_hj", "_chz"])
    print(hj_chz_hosts[["hostname", "pl_name_hj", "pl_name_chz"]])
    print("--------------------------------------------------------------------------------")

    hj_chz_pl = pd.DataFrame({"pl-name": list(set(hj_chz_hosts["pl_name_hj"]).union(
            set(hj_chz_hosts["pl_name_chz"])))})

    # load alfven data without planet names
    ashc = pd.read_csv("tables-merged/alfven_data.csv")
    ashc_hj_chz = pd.merge(ashc, hj_chz_pl, how="right", on="pl-name")
    # TODO drop index ?
    ashc_hj_chz.drop(columns="index", inplace=True)
    ashc_hj_chz.dropna(subset="hostname", inplace=True)
    ashc_hj_chz.sort_values(by=["hostname", "pl-letter"], inplace=True)
    print(ashc_hj_chz[["hostname", "pl-name", "ASHC", "CHZ", "OHZ",
                       "pl-orbsmax", "pl-orbeccen", "pl-bmasse", "pl-bmassj",
                       "Prot", "Ro", "st-spectype", "st-teff", "st-mass"]])
