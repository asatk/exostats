import os
import pandas as pd
import pyvo as vo
import re

from aliases import pool_aliases

# create object that will service TAP queries to NEA TAP server
url_NEA = "https://exoplanetarchive.ipac.caltech.edu/TAP"
service = vo.dal.TAPService(url_NEA)

#------------------------------------------------------------------------------
# COLUMNS TO FETCH
#------------------------------------------------------------------------------

# planetary systems (NEA confirmed exoplanets)
ps_cols = [
    "pl_name", "hostname",
    "st_mass", "st_masserr1", "st_masserr2",
    "sy_vmag", "sy_vmagerr1", "sy_vmagerr2",
    "sy_kmag", "sy_kmagerr1", "sy_kmagerr2",
    "pl_orbsmax", "pl_orbsmaxerr1", "pl_orbsmaxerr2",
    "pl_orbeccen", "pl_orbeccenerr1", "pl_orbeccenerr2",
    "sy_umag", "sy_umagerr1", "sy_umagerr2",
    "st_teff", "st_tefferr1", "st_tefferr2",
    "st_lum", "st_lumerr1", "st_lumerr2",
    "sy_dist", "sy_disterr1", "sy_disterr2",
    "pl_bmasse", "pl_bmasseerr1", "pl_bmasseerr2",
    "st_age", "st_ageerr1", "st_ageerr2",
    "pl_orbsmax", "pl_orbsmaxerr1", "pl_orbsmaxerr2"
]

# kepler planets and candidates
koi_cols = [
    "kepid", "koi_disposition",
    "koi_smass", "koi_smass_err1", "koi_smass_err2",
    "koi_steff", "koi_steff_err1", "koi_steff_err2",
    "koi_srad", "koi_srad_err1", "koi_srad_err2",
    "koi_gmag", "koi_gmag_err",
    "koi_rmag", "koi_rmag_err",
    "koi_kmag", "koi_kmag_err",
    "koi_sma", "koi_sma_err1", "koi_sma_err2",
    "koi_eccen", "koi_eccen_err1", "koi_eccen_err2",
]

# k2 planets and candidates
k2_cols = [
    "pl_name", "hostname", "disposition", "k2_name",
    "st_mass", "st_masserr1", "st_masserr2",
    "sy_vmag", "sy_vmagerr1", "sy_vmagerr2",
    "sy_kmag", "sy_kmagerr1", "sy_kmagerr2",
    "pl_orbsmax", "pl_orbsmaxerr1", "pl_orbsmaxerr2",
    "pl_orbeccen", "pl_orbeccenerr1", "pl_orbeccenerr2",
    "sy_umag", "sy_umagerr1", "sy_umagerr2",
    "st_teff", "st_tefferr1", "st_tefferr2",
    "sy_dist", "sy_disterr1", "sy_disterr2"
]

# tess planets and candidates
toi_cols = [
    "tid", "toipfx",
    "st_teff", "st_tefferr1", "st_tefferr2",
    "st_rad", "st_raderr1", "st_raderr2",
    "pl_orbper", "pl_orbpererr1", "pl_orbpererr2",
]



def fetch_table(table_name: str, cols: list[str], cond_str: str=None):
    """
    Fetch the data for a named table in the NEA table list using the Table
    Access Protocol (TAP).
    """

    # combine list of table columns as one string separated by commas
    cols_str = ",".join(cols)

    # ADQL query
    query = f"""
        SELECT {cols_str}
        FROM {table_name}
        """

    # send TAP request to VO server
    resp = service.search(query)

    # failed query ends process
    print(f"Successful query for table `{table_name}`")

    # extract data from response as pd.DataFrame object
    table = resp.to_table()
    table = table.to_pandas()
    return table



def update_aliases(hostnames: pd.Series, tag_list: list[str], fname: str):
    """
    Update aliases table for the specified hostnames and tags/catalogs
    """

    if os.path.isfile(fname_al):
        al = pd.read_csv(fname_al)
        al_tags = set(al.columns)
        al_hosts = set(al["hostname"])
        
        tags_set = set(tag_list)
        hosts_set = set(hostnames)

        tags_diff = set(tag_list) - al_tags
        hosts_diff = set(hostnames) - al_hosts

        # same tags but new hosts -- append new rows
        if len(tags_diff) == 0 and len(hosts_diff) > 0:
            print("Tags are the same but the hosts differ -- adding new rows")
            hostnames = pd.Series(list(hosts_diff))
            al_new = pool_aliases(hostnames, tag_list)
            al = pd.concat([al, al_new], ignore_index=True)

        # new tags but same hosts -- merge new columns
        elif len(tags_diff) > 0 and len(hosts_diff) == 0:
            print("Tags differ but the hosts are the same -- get new aliases")
            tag_list = list(tags_diff)
            al_new = pool_aliases(hostnames, tag_list)
            al = pd.merge(al, al_new, on="hostname", how="inner")

        # new tags and new hosts -- get all data
        elif len(tags_diff) > 0 or len(hosts_diff) > 0:
            print("Tags differ and there are new hosts -- getting new data")
            tag_list = list(tags_diff)
            hostnames = pd.Series(list(hosts_diff))
            al = pool_aliases(hostnames, tag_list)
        else:
            print("File on disk is up to date!")

    else:
        # file doesn't exist on disk -- get all data
        print("No aliases file exists -- getting new data")
        al = pool_aliases(hostnames, tag_list)

    return al



if __name__ == "__main__":

    ##### REQUEST TABLES

    # fetch tables from NEA using TAP
    ps = fetch_table("pscomppars", ps_cols)
    koi = fetch_table("cumulative", koi_cols)
    k2 = fetch_table("k2pandc", k2_cols)
    toi = fetch_table("toi", toi_cols)



    ##### PSCP ALIASES

    # load list of exoplanet host star names
    hostnames = pd.Series(pd.unique(ps["hostname"]), name="hostname")

    # fetch aliases from NEA using API endpoint or existing file on disk
    fname_al = "../db/nea/aliases.csv"
    tag_list = ["KOI", "KIC", "EPIC", "TOI", "TIC", "Gaia DR2", "Gaia DR3"]
    al = update_aliases(hostnames, tag_list, fname_al)

    # add aliases of confirmed exoplanets to PSCP table
    ps = pd.merge(ps, al, how="left", on="hostname")

    # stars lacking alias from a given catalog receive a -1 as integer placeholder value
    ps[tag_list] = ps[tag_list].fillna(-1)
    ps[tag_list] = ps[tag_list].astype(int)



    ##### SAVE TABLES

    # check existence of directory for NEA tables
    if not os.path.isdir("../db/nea/"):
        os.mkdir("../db/nea/")

    # write tables to db/nea/ directory
    ps.to_csv("../db/nea/ps.csv", index=False)
    koi.to_csv("../db/nea/koi.csv", index=False)
    k2.to_csv("../db/nea/k2.csv", index=False)
    toi.to_csv("../db/nea/toi.csv", index=False)
    al.to_csv("../db/nea/aliases.csv", index=False)

