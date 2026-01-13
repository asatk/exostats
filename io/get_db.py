from datetime import datetime
import numpy as np
import pyvo as vo
import re



#------------------------------------------------------------------------------
# COLUMNS TO FETCH
#------------------------------------------------------------------------------

# Planetary Systems Composite Data (NEA confirmed exoplanets)
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

# Kepler planets and candidates
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

# K2/EPIC planets and candidates
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

# TESS planets and candidates
toi_cols = [
    "tid", "toipfx",
    "st_teff", "st_tefferr1", "st_tefferr2",
    "st_rad", "st_raderr1", "st_raderr2",
    "pl_orbper", "pl_orbpererr1", "pl_orbpererr2",
]



if __name__ == "__main__":

    ##### NEA CONFIRMED AND CANDIDATE HOSTS

    # create object that will service TAP queries to NEA TAP server
    url_NEA = "https://exoplanetarchive.ipac.caltech.edu/TAP"
    service_NEA = vo.dal.TAPService(url_NEA)

    # ADQL query sent to NEA TAP service
    query_NEA = """
        SELECT hostname, 1 AS disposition
        FROM exo_tap.pscomppars
        UNION
        SELECT hostname, 1 AS disposition
        FROM exo_tap.k2pandc
        WHERE disposition = 'CONFIRMED'
        UNION
        SELECT hostname, 0 AS disposition
        FROM exo_tap.k2pandc
        WHERE disposition = 'CANDIDATE'
        UNION
        SELECT 'KIC ' || CAST(cumulative.kepid AS VARCHAR(8)) AS hostname, 1 AS disposition
        FROM exo_tap.cumulative
        WHERE koi_disposition = 'CONFIRMED'
        UNION
        SELECT 'KIC ' || CAST(cumulative.kepid AS VARCHAR(8)) AS hostname, 0 AS disposition
        FROM exo_tap.cumulative
        WHERE koi_disposition = 'CANDIDATE'
        UNION
        SELECT 'TIC ' || CAST(toi.tid AS VARCHAR(11)) AS hostname, 1 AS disposition
        FROM exo_tap.toi
        WHERE tfopwg_disp = 'CP' or tfopwg_disp = 'KP'
        UNION
        SELECT 'TIC ' || CAST(toi.tid AS VARCHAR(11)) AS hostname, 0 AS disposition
        FROM exo_tap.toi
        WHERE tfopwg_disp = 'PC' or tfopwg_disp = 'APC'
    """
    time = datetime.now()
    res_NEA = service_NEA.run_sync(query_NEA, maxrec=100_000)
    dt = datetime.now() - time
    hostnames = res_NEA.to_table().to_pandas()
    hostnames.to_csv("../db/hostnames.csv",index=False)
    print(f"Exoplanet Host Names Identified ({int(dt.seconds/60)}m {dt.seconds%60}s)")



    ##### SIMBAD ALIASES

    # create object that will service TAP queries to SIMBAD TAP server
    url_SIMBAD = "https://simbad.unistra.fr/simbad/sim-tap"
    service_SIMBAD = vo.dal.TAPService(url_SIMBAD)

    # ADQL query sent to SIMBAD TAP service
    query_SIMBAD = """
        SELECT basic.main_id, tbl.disposition, id1.id AS kic, id2.id AS tic, id3.id AS epic, id4.id AS gaia_dr3_id, tbl.oidref
        FROM (
          SELECT MAX(disposition) AS disposition, oidref
          FROM TAP_UPLOAD.hostnames
          INNER JOIN ident as id1
          ON id1.id = hostnames.hostname
          GROUP BY oidref
        ) AS tbl
        LEFT JOIN (
          SELECT ident.id, ident.oidref
          FROM ident
          WHERE ident.id LIKE 'KIC %'
        ) AS id1
        ON tbl.oidref = id1.oidref
        LEFT JOIN (
          SELECT ident.id, ident.oidref
          FROM ident
          WHERE ident.id LIKE 'TIC %'
        ) AS id2
        ON tbl.oidref = id2.oidref
        LEFT JOIN (
          SELECT ident.id, ident.oidref
          FROM ident
          WHERE ident.id LIKE 'EPIC %'
        ) AS id3
        ON tbl.oidref = id3.oidref
        LEFT JOIN (
          SELECT ident.id, ident.oidref
          FROM ident
          WHERE ident.id LIKE 'Gaia DR3 %'
        ) AS id4
        ON tbl.oidref = id4.oidref
        LEFT JOIN basic
        ON tbl.oidref = basic.oid;
    """

    time = datetime.now()
    res_SIMBAD = service_SIMBAD.run_sync(query_SIMBAD, maxrec=100_000, uploads={"hostnames": res_NEA})
    dt = datetime.now() - time
    aliases = res_SIMBAD.to_table().to_pandas()
    print(f"Aliases Found ({int(dt.seconds/60)}m {dt.seconds%60}s)")



    ##### POST-PROCESSING

    cols = ["oidref", "tic", "kic", "epic", "gaia_dr3_id"]
    aliases[cols] = aliases[cols].fillna(-1)

    p = re.compile(r"\w+\s+(\d+)$")
    def id_to_num(r: str):
        row = []
        for s in r:
            if s is np.nan or s is None:
                row.append(-1)
                continue

            if type(s) is int:
                row.append(s)
                continue
            
            m = p.search(s)
            if m is None:
                row.append(-1)
            else:
                row.append(int(m[1]))

        return row

    # strip catalog name from identifiers to convert into integers
    aliases[cols] = aliases[cols].apply(id_to_num, axis=1, result_type="expand")

    # remove dupes (mostly in TIC) for same object
    aliases = aliases.groupby(by="oidref").head(1)



    ##### SAVE TABLE AND SUMMARY

    # write tables to db/nea/ directory
    aliases.to_csv("../db/aliases.csv", index=False)

    print(aliases.count())
