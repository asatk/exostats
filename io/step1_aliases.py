"""
Name all confirmed and candidate planets in the Nasa Exoplanet Archive (NEA)
database. Merges data from Kepler, K2, and TESS input catalogs in addition to
other discoveries listed in the NEA's Planetary Systems Composite Parameters
table. Each planet is listed with its host's name.

Author: Anthony Atkinson
Last modified: 2026.06.26
"""

from datetime import datetime
import numpy as np
import pyvo as vo
import re



if __name__ == "__main__":

    # ----------------------- #
    # NEA CONF. & CAND. HOSTS #
    # ----------------------- #

    # dispatch queries to NEA TAP server
    url_NEA = "https://exoplanetarchive.ipac.caltech.edu/TAP"
    service_NEA = vo.dal.TAPService(url_NEA)

    # ADQL query sent to NEA TAP service to fetch planet and host names
    query_NEA = r"""
SELECT hostname
FROM pscomppars
UNION (
  SELECT hostname
  FROM k2pandc
  WHERE disposition = 'CONFIRMED' OR disposition = 'CANDIDATE'
) UNION (
  SELECT 'KIC ' || CAST(cumulative.kepid AS VARCHAR(8)) AS hostname
  FROM cumulative
  WHERE koi_disposition = 'CONFIRMED' OR koi_disposition = 'CANDIDATE'
) UNION (
  SELECT 'TIC ' || CAST(toi.tid AS VARCHAR(11)) AS hostname
  FROM toi
  WHERE tfopwg_disp = 'CP' OR tfopwg_disp = 'KP' OR tfopwg_disp = 'PC' OR tfopwg_disp = 'APC'
)"""

    # submit and time query
    time = datetime.now()
    res_NEA = service_NEA.run_sync(query_NEA, maxrec=100_000)
    dt = datetime.now() - time

    # convert XML/VOTable response -> astropy.table.Table -> pd.DataFrame
    hostnames = res_NEA.to_table().to_pandas()

    # save hostnames
    hostnames.to_csv("../db/hostnames.csv",index=False)

    print(f"{hostnames['hostname'].count()} Exoplanet Host Names Identified (in {int(dt.seconds/60)}m {dt.seconds%60}s)")



    # -------------- #
    # SIMBAD ALIASES #
    # -------------- #

    # dispatch queries to SIMBAD TAP server
    url_SIMBAD = "https://simbad.unistra.fr/simbad/sim-tap"
    service_SIMBAD = vo.dal.TAPService(url_SIMBAD)

    # ADQL query sent to SIMBAD TAP service
    query_SIMBAD = \
"""
SELECT tbl.hostname AS hostname, tbl.oidref, id1.id AS kic, id2.id AS tic, id3.id AS epic, id4.id AS gaia_dr3_id
FROM (
  SELECT hostname, oidref
  FROM TAP_UPLOAD.hostnames
  INNER JOIN ident AS id1
  ON id1.id = hostnames.hostname
  GROUP BY oidref, hostname
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
ON tbl.oidref = id4.oidref;"""

    # submit and time query
    time = datetime.now()
    res_SIMBAD = service_SIMBAD.run_sync(query_SIMBAD, maxrec=100_000, uploads={"hostnames": res_NEA})
    dt = datetime.now() - time

    # convert XML/VOTable response -> astropy.table.Table -> pd.DataFrame
    aliases = res_SIMBAD.to_table().to_pandas()
    print(f"{aliases['hostname'].count()} Aliases Found (in {int(dt.seconds/60)}m {dt.seconds%60}s)")



    # ------------------- #
    # ID STRING TO NUMBER #
    # ------------------- #


    # names of ID columns to isolate integer ID num from ID string
    cols = ["oidref", "tic", "kic", "epic", "gaia_dr3_id"]

    # indicate absence in catalog with ID = -1
    aliases[cols] = aliases[cols].fillna(-1)

    # match pattern: '<catalog> <number>' to each ID if present
    p = re.compile(r"\w+\s+(\d+)$")
    def id_to_num(r: str):
        row = []
        for s in r:

            # null value ID
            if s is np.nan or s is None:
                row.append(-1)
                continue

            # ID is already an integer (or already -1)
            if type(s) is int:
                row.append(s)
                continue

            # match ID string to pattern: '<catalog> <number>'
            m = p.search(s)

            # no match, no ID
            if m is None:
                row.append(-1)
            # match present, add ID num to row in corresponding ID column
            else:
                row.append(int(m[1]))

        return row

    # strip catalog name from identifiers to convert into integers
    aliases[cols] = aliases[cols].apply(id_to_num, axis=1, result_type="expand")

    # remove dupes (mostly in TIC) for same object
    aliases = aliases.drop_duplicates("oidref")



    # ---------------------- #
    # TABLE EXPORT + SUMMARY #
    # ---------------------- #

    # write tables to db/ directory
    aliases.to_csv("../db/aliases.csv", index=False)

    # summarize identified planets (conf+cand)
    print(aliases.count())
