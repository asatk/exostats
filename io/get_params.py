import pandas as pd
import pyvo as vo

hosts = pd.read_csv("../db/prot.csv")

# create object that will service TAP queries to SIMBAD TAP server
url_SIMBAD = "https://simbad.unistra.fr/simbad/sim-tap"
service_SIMBAD = vo.dal.TAPService(url_SIMBAD)

# ADQL query sent to SIMBAD TAP service
query_SIMBAD = """
    SELECT oidref, mesFe_H.teff AS teff, mesFe_H.fe_h AS fe_h
    FROM mesFe_H
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

res_SIMBAD = service_SIMBAD.run_sync(query_SIMBAD, maxrec=100_000, uploads={"hosts": hosts})

