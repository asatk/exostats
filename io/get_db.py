from datetime import datetime
import numpy as np
import pyvo as vo
import re



if __name__ == "__main__":

    ##### NEA CONFIRMED AND CANDIDATE HOSTS

    # create object that will service TAP queries to NEA TAP server
    url_NEA = "https://exoplanetarchive.ipac.caltech.edu/TAP"
    service_NEA = vo.dal.TAPService(url_NEA)

    # ADQL query sent to NEA TAP service
    query_NEA = """
        SELECT hostname AS nea_id, 1 AS disposition
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
        SELECT tbl.nea_id AS NEA_id, tbl.oidref, tbl.disposition, id1.id AS kic, id2.id AS tic, id3.id AS epic, id4.id AS gaia_dr3_id
        FROM (
          SELECT MAX(disposition) AS disposition, nea_id, oidref
          FROM TAP_UPLOAD.hostnames
          INNER JOIN ident as id1
          ON id1.id = hostnames.nea_id
          GROUP BY oidref, nea_id
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
        ON tbl.oidref = id4.oidref;
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
    aliases = aliases.drop_duplicates("oidref")



    ##### SAVE TABLE AND SUMMARY

    # write tables to db/nea/ directory
    aliases.to_csv("../db/aliases.csv", index=False)

    print(aliases.count())
