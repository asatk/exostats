from astropy.table import Table
import numpy as np
import pandas as pd
import pyvo as vo


#### NEA PARAMS






def choose_msmt(msmts):
    # if len(msmts) % 2 == 1:
    #     return np.median(msmts)

    mean = np.mean(msmts)
    choose_idx = np.argmin(msmts - mean)

    return msmts[choose_idx]


simbad_query_template = \
"""
SELECT %s
FROM TAP_UPLOAD.hosts AS tbl
JOIN %s
ON tbl.oidref = %s.oidref
"""

def build_simbad_query(table_name: str, *params: list[str]|str):
    if isinstance(params, list):
        params = ",".join(params)

    s = simbad_query_template.format(params, table_name, table_name)
    return s


#------------------------------------------------------------------------------
# COLUMNS TO FETCH
#------------------------------------------------------------------------------

# TODO use PS not PSCP for get_params
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


# TODO combine with NEA catalog

# create object that will service TAP queries to SIMBAD TAP server
url_SIMBAD = "https://simbad.unistra.fr/simbad/sim-tap"
service_SIMBAD = vo.dal.TAPService(url_SIMBAD)

# ADQL query sent to SIMBAD TAP service
query1 = \
"""
SELECT tbl.oidref, teff, fe_h, prot
FROM TAP_UPLOAD.hosts AS tbl
LEFT JOIN mesfe_h
ON tbl.oidref = mesfe_h.oidref
LEFT JOIN
    (SELECT oidref, period AS prot
    FROM mesvar
    WHERE vartyp = 'ROT') AS vartbl
ON tbl.oidref = vartbl.oidref;
"""

query2 = \
"""
SELECT tbl.oidref, teff, fe_h, prot,
    flux, V AS vmag, K AS kmag, F200W AS nuvmag
FROM TAP_UPLOAD.hosts AS tbl
LEFT JOIN allfluxes
ON tbl.oidref = allfluxes.oidref
LEFT JOIN flux
ON tbl.oidref = flux.oidref;
"""



# ADQL query sent to SIMBAD TAP service
query3 = \
"""
SELECT tbl.oidref, tbl.teff, tbl.fe_h, prot,
    flux, vmag, kmag, nuvmag,
    dist, diameter / 2 AS rad, vsini
FROM TAP_UPLOAD.hosts AS tbl
LEFT JOIN mesdistance
ON tbl.oidref = mesdistance.oidref
LEFT JOIN mesdiameter
ON tbl.oidref = mesdiameter.oidref
LEFT JOIN mesrot
ON tbl.oidref = mesrot.oidref;
"""



# Query for uncertainties
query4 = \
"""
SELECT tbl.oidref, teff, fe_h, prot,
    tbl.flux, flux_err, vmag, kmag, nuvmag,
    tbl.dist, minus_err AS dist_err1, plus_err AS dist_err2,
    rad, error / 2 AS rad_err, tbl.vsini, vsini_err 
FROM TAP_UPLOAD.hosts AS tbl
LEFT JOIN flux
ON tbl.oidref = flux.oidref
LEFT JOIN mesdistance
ON tbl.oidref = mesdistance.oidref
LEFT JOIN mesdiameter
ON tbl.oidref = mesdiameter.oidref
LEFT JOIN mesrot
ON tbl.oidref = mesrot.oidref;
"""



if __name__ == "__main__":

    hosts = pd.read_csv("../db/aliases.csv")
    hosts = Table.from_pandas(hosts)

    # TODOs
    # match uncertainties with measurements
    # get msmt using closest to mean OR smallest unc

    res1 = service_SIMBAD.run_sync(query1, maxrec=200_000, uploads={"hosts": hosts})
    params_temp = res1.to_table().to_pandas().groupby("oidref").median().reset_index()
    params_temp = Table.from_pandas(params_temp)
    print("req1")

    res2 = service_SIMBAD.run_sync(query2, maxrec=200_000, uploads={"hosts": params_temp})
    params_temp = res2.to_table().to_pandas().groupby("oidref").median().reset_index()
    params_temp = Table.from_pandas(params_temp)
    print("req2")

    res3 = service_SIMBAD.run_sync(query3, maxrec=200_000, uploads={"hosts": params_temp})
    params_temp = res3.to_table().to_pandas().groupby("oidref").median().reset_index()
    params_temp = Table.from_pandas(params_temp)
    print("req3")

    res4 = service_SIMBAD.run_sync(query4, maxrec=500_000, uploads={"hosts": params_temp})
    params = res4.to_table().to_pandas().groupby("oidref").max().reset_index()
    print("req4")

    params.to_csv("../db/params.csv", index=False)