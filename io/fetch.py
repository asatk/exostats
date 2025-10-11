import os
import pyvo as vo

url_NEA = "https://exoplanetarchive.ipac.caltech.edu/TAP"
service = vo.dal.TAPService(url_NEA)

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

    # extract data from response as Astropy.table.Table object
    table = resp.to_table()
    return table



if __name__ == "__main__":
    # create object that will service TAP queries to NEA TAP server
    service = vo.dal.TAPService(url_NEA)

    # system params to fetch from each table

    # planetary systems (all confirmed)
    ps_cols = [
        "pl_name", "hostname", "tic_id",
        "st_mass", "st_masserr1", "st_masserr2",
        "sy_vmag", "sy_vmagerr1", "sy_vmagerr2",
        "sy_kmag", "sy_kmagerr1", "sy_kmagerr2",
        "pl_orbsmax", "pl_orbsmaxerr1", "pl_orbsmaxerr2",
        "pl_orbeccen", "pl_orbeccenerr1", "pl_orbeccenerr2",

    ]

    # kepler planets and candidates
    koi_cols = [
        "kepid", "koi_disposition",
        "koi_smass", "koi_smass_err1", "koi_smass_err2",
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
    ]

    # tess planets and candidates
    toi_cols = [
        "tid", "toipfx",
        "st_teff", "st_tefferr1", "st_tefferr2",
        "st_rad", "st_raderr1", "st_raderr2",
        "pl_orbper", "pl_orbpererr1", "pl_orbpererr2",
    ]

    # fetch tables from NEA using TAP
    tbl_ps = fetch_table("pscomppars", ps_cols)
    tbl_koi = fetch_table("cumulative", koi_cols)
    tbl_k2 = fetch_table("k2pandc", k2_cols)
    tbl_toi = fetch_table("toi", toi_cols)

    # check existence of directory for NEA tables
    if not os.path.isdir("../db/nea/"):
        os.mkdir("../db/nea/")

    # write tables to db/nea/ directory
    tbl_ps.write("../db/nea/ps.csv", overwrite=True)
    tbl_koi.write("../db/nea/koi.csv", overwrite=True)
    tbl_k2.write("../db/nea/k2.csv", overwrite=True)
    tbl_toi.write("../db/nea/toi.csv", overwrite=True)

