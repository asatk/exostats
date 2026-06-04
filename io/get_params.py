from statistics import median_low

from astropy.table import Table
import numpy as np
import pandas as pd
import pyvo as vo


#### NEA PARAMS


def median_low(vals):
    if len(vals) % 2 == 0:
        return np.median(vals[:-1])
    return np.median(vals)


def choose_msmt(df, col):
    df_grp = df.groupby("oidref")[[col]]
    return df_grp.apply(median_low)
    # if len(msmts) % 2 == 1:
    #     return np.median(msmts)

    # mean = np.mean(msmts)
    # choose_idx = np.argmin(msmts - mean)
    #
    # return msmts[choose_idx]



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


simbad_query_template = \
"""
SELECT {0}
FROM TAP_UPLOAD.hosts AS tbl
LEFT JOIN {1}
ON tbl.oidref = {1}.oidref;
"""



def build_simbad_query(table_name: str, *params: list[str]|str):
    if isinstance(params, list):
        params = ",".join(params)

    s = simbad_query_template.format(params, table_name, table_name)
    return s

def form_query(t_name, p_names):
    cols = ["tbl.oidref"]
    if isinstance(p_names, list):
        cols.extend(p_names)
    else:
        cols.append(p_names)

    cols = ", ".join(cols)

    qs = simbad_query_template.format(cols, t_name)
    return qs


def submit_query(qs: str, hosts: Table):
    # job = service_SIMBAD.submit_job(s, maxrec=200_000, uploads={"hosts": hosts})
    print(qs)
    res = service_SIMBAD.run_sync(qs, maxrec=200_000, uploads={"hosts": hosts}).to_table().to_pandas()
    res.rename({"tbl.oidref": "oidref"}, inplace=True, axis=1)
    res.set_index("oidref", inplace=True)
    print(res)
    print(f"{len(res)} records returned.")
    return res

qs_sptype = \
"""
SELECT tbl.oidref, sptype
FROM TAP_UPLOAD.hosts AS tbl
LEFT JOIN messpt
ON tbl.oidref = messpt.oidref
WHERE sptype != '';
"""

# TODO had to upload this to the old TAP interface and extract data from there
# SELECT tbl.col2, sptype
# FROM TAP_UPLOAD.hosts as tbl
# LEFT JOIN messpt
# ON tbl.col2 = messpt.oidref
# WHERE sptype != '';


qs_prot = \
"""
SELECT tbl.oidref, period prot
FROM TAP_UPLOAD.hosts AS tbl
LEFT JOIN mesvar
ON tbl.oidref = mesvar.oidref
WHERE vartyp = 'ROT';
"""

# Query for uncertainties
query_errs = \
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

    hosts_df = pd.read_csv("../db/aliases.csv")
    hosts = Table.from_pandas(hosts_df)
    hosts_df.set_index(["oidref"], inplace=True)

    # TODOs
    # match uncertainties with measurements
    # get msmt using closest to mean OR smallest unc
    jobs = []

    # res_spt = submit_query(qs_sptype, hosts=hosts)
    res_spt = pd.read_csv("../db/sptype.csv")
    res_teff = submit_query(form_query("mesfe_h", "teff"), hosts=hosts)
    # res_met = submit_query(form_query("mesfe_h", "fe_h AS met"), hosts=hosts)
    # res_prot = submit_query(qs_prot, hosts=hosts)
    # res_flux = submit_query(form_query("flux", "flux"), hosts=hosts)
    # res_vmag = submit_query(form_query("allfluxes", "V as Vmag"), hosts=hosts)
    # res_kmag = submit_query(form_query("allfluxes", "K as kmag"), hosts=hosts)
    # res_nuvmag = submit_query(form_query("allfluxes", "F200W as nuvmag"), hosts=hosts)
    # res_dist = submit_query(form_query("mesdistance", "dist"), hosts=hosts)
    # res_rad = submit_query(form_query("mesdiameter", "diameter / 2 AS rad"), hosts=hosts)
    # res_vsini = submit_query(form_query("mesrot", "vsini"), hosts=hosts)

    spt = res_spt.groupby("oidref")[["sptype"]].apply(lambda l: l.head(1))
    teff = choose_msmt(res_teff, "teff")
    # met = choose_msmt(res_met, "met")
    # prot = choose_msmt(res_prot, "prot")
    # flux = choose_msmt(res_flux, "flux")
    # vmag = choose_msmt(res_vmag, "vmag")
    # kmag = choose_msmt(res_kmag, "kmag")
    # nuvmag = choose_msmt(res_nuvmag, "nuvmag")
    # dist = choose_msmt(res_dist, "dist")
    # rad = choose_msmt(res_rad, "rad")
    # vsini = choose_msmt(res_vsini, "vsini")

    params = [
        spt,
        teff,
        # met,
        # prot,
        # flux,
        # vmag,
        # kmag,
        # nuvmag,
        # dist,
        # rad,
        # vsini
    ]

    df = hosts_df.copy()
    for param in params:
        print(param)
        df = pd.merge(df, param, on="oidref", how="outer")

    # df = pd.concat([hosts_df,
    #                 spt,
    #                 teff,
    #                 met,
    #                 # prot,
    #                 # flux,
    #                 # vmag,
    #                 # kmag,
    #                 # nuvmag,
    #                 # dist,
    #                 # rad,
    #                 # vsini
    #                 ], join="outer")
    df.reset_index(inplace=True, drop=False)
    print(df.head(20))
    df.to_csv("../db/params-new.csv", index=False)


    # jobs.append(job_spt)
    #
    # i = 0
    # while len(jobs) > 0:
    #     status = jobs[i].phase
    #     if status == "COMPLETED":
    #
    #     i += 1


    #
    # print(query1)
    #
    # res1 = service_SIMBAD.run_sync(query1, maxrec=200_000, uploads={"hosts": hosts})
    # params_temp = res1.to_table().to_pandas().groupby("oidref").median().reset_index()
    # params_temp = Table.from_pandas(params_temp)
    # print("req1")
    #
    # res2 = service_SIMBAD.run_sync(query2, maxrec=200_000, uploads={"hosts": params_temp})
    # params_temp = res2.to_table().to_pandas().groupby("oidref").median().reset_index()
    # params_temp = Table.from_pandas(params_temp)
    # print("req2")
    #
    # res3 = service_SIMBAD.run_sync(query3, maxrec=200_000, uploads={"hosts": params_temp})
    # params_temp = res3.to_table().to_pandas().groupby("oidref").median().reset_index()
    # params_temp = Table.from_pandas(params_temp)
    # print("req3")
    #
    # res4 = service_SIMBAD.run_sync(query4, maxrec=500_000, uploads={"hosts": params_temp})
    # params = res4.to_table().to_pandas().groupby("oidref").max().reset_index()
    # print("req4")

    # params.to_csv("../db/params.csv", index=False)