import warnings

from astropy.table import Table
import numpy as np
import pandas as pd
import pyvo as vo
import re
from time import sleep

from pyvo.dal import AsyncTAPJob

from ioutil import concat_update_df, merge_update_df



#------------------------------------------------------------------------------
# COLUMNS TO FETCH
#------------------------------------------------------------------------------

# Planetary Systems Composite Data (NEA confirmed exoplanets)
cols_pscomppars = [
    "hostname",
    "pl_name",
    "pl_letter",
    "pl_orbsmax",
    "pl_orbincl",
    "pl_orbeccen",
    "pl_bmasse",
    "pl_rade",
    "st_teff",
    "st_met",
    "st_lum",
    "st_age",
    "st_mass",
    "st_rad",
    "st_spectype",
    "st_rotp",
    "st_vsin",
    "sy_dist",
    "sy_vmag",
    "sy_kmag",
    "sy_snum",
    "sy_pnum"
]

# Kepler planets and candidates
cols_cumulative = [
    "kepid",
    "kepler_name AS pl_name",
    "koi_sma AS pl_orbsmax",
    "koi_incl AS pl_orbincl",
    "koi_eccen AS pl_orbeccen",
    "koi_prad AS pl_rade",
    "koi_steff AS st_teff",
    "koi_smet AS st_met",
    "koi_sage AS st_age",
    "koi_smass AS st_mass",
    "koi_srad AS st_rad",
    "koi_kmag AS sy_kmag",
    "koi_gmag AS sy_gmag",
    "koi_rmag AS sy_rmag",
    "koi_kepmag AS sy_kepmag",
    "koi_srho AS st_rho"
]

# K2/EPIC planets and candidates
cols_k2pandc = [
    "hostname",
    "pl_name",
    "pl_letter",
    "pl_orbsmax",
    "pl_orbincl",
    "pl_orbeccen",
    "pl_bmasse",
    "pl_rade",
    "st_teff",
    "st_met",
    "st_lum",
    "st_age",
    "st_mass",
    "st_rad",
    "st_spectype",
    "st_rotp",
    "st_vsin",
    "sy_dist",
    "sy_vmag",
    "sy_kmag",
    "sy_snum",
    "sy_pnum"
]

# TESS planets and candidates
cols_toi = [
    "tid",
    "toidisplay AS pl_name",
    "pl_rade",
    "st_teff",
    "st_rad",
    "st_dist AS sy_dist",
    "pl_orbper"
]



#------------------------------------------------------------------------------
# TAP QUERY SETUP
#------------------------------------------------------------------------------

url_NEA = "https://exoplanetarchive.ipac.caltech.edu/TAP"
service_NEA = vo.dal.TAPService(url_NEA)

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
ON tbl.oidref = {1}.oidref
"""

# TODO understand MESPOS place in SIMBAD
# will only have one msmt per host because of MESPOS
qs_spt = \
r"""
SELECT tbl.oidref, sptype AS st_spectype
FROM TAP_UPLOAD.hosts as tbl
LEFT JOIN messpt
ON tbl.oidref = messpt.oidref
WHERE regexp(sptype, '^[\\u0000-\\u007f]+$') = 1 AND mespos = 1;
"""

qs_prot = \
"""
SELECT tbl.oidref, period AS st_rotp
FROM TAP_UPLOAD.hosts AS tbl
LEFT JOIN mesvar
ON tbl.oidref = mesvar.oidref
WHERE vartyp = 'ROT' AND period IS NOT NULL;
"""

qs_dist_pc = \
"""
SELECT tbl.oidref, dist AS sy_dist_pc
FROM TAP_UPLOAD.hosts AS tbl
LEFT JOIN mesdistance
ON tbl.oidref = mesdistance.oidref
WHERE dist IS NOT NULL AND unit = 'pc';
"""

qs_dist_kpc = \
"""
SELECT tbl.oidref, dist * 1000 AS sy_dist_kpc
FROM TAP_UPLOAD.hosts AS tbl
LEFT JOIN mesdistance
ON tbl.oidref = mesdistance.oidref
WHERE dist IS NOT NULL AND unit = 'kpc';
"""

qs_dist_mpc = \
"""
SELECT tbl.oidref, dist * 1000000 AS sy_dist_mpc
FROM TAP_UPLOAD.hosts AS tbl
LEFT JOIN mesdistance
ON tbl.oidref = mesdistance.oidref
WHERE dist IS NOT NULL AND unit = 'Mpc';
"""

# TODO email CDS/SIMBAD about lack of UNION kw
"""
SELECT tbl1.oidref, dist AS sy_dist
FROM TAP_UPLOAD.hosts AS tbl1
LEFT JOIN (
  SELECT oidref, dist as sy_dist
  FROM mesdistance
  WHERE dist IS NOT NULL AND unit = 'pc'
  UNION (
    SELECT oidref, dist / 1000 as sy_dist
    FROM mesdistance
    WHERE dist IS NOT NULL AND unit = 'kpc')
  UNION (
    SELECT oidref, dist / 1000000 as sy_dist
    FROM mesdistance
    WHERE dist IS NOT NULL AND unit = 'Mpc')
  ) AS mesd
ON tbl1.oidref = mesd.oidref;
"""

qs_rad = \
"""
SELECT tbl.oidref, diameter / 2 / 695700 AS st_rad
FROM TAP_UPLOAD.hosts AS tbl
LEFT JOIN mesdiameter
ON tbl.oidref = mesdiameter.oidref
WHERE diameter IS NOT NULL AND unit = 'km';
"""

# NEA queries

query_pscomppars = rf"""
SELECT {', '.join(cols_pscomppars)}, 1 AS disposition, 'pscp' AS db
FROM pscomppars"""

query_cumulative = rf"""
SELECT {', '.join(cols_cumulative)}, 0 AS disposition, 'koi' AS db
FROM cumulative
WHERE koi_disposition = 'CANDIDATE'
UNION (
  SELECT {','.join(cols_cumulative)}, 1 AS disposition, 'koi' AS db
  FROM cumulative
  WHERE koi_disposition = 'CONFIRMED'
)"""

query_k2pandc = rf"""
SELECT {', '.join(cols_k2pandc)}, 0 AS disposition, 'k2' AS db
FROM k2pandc
WHERE disposition = 'CANDIDATE' AND default_flag = 1
UNION (
  SELECT {','.join(cols_k2pandc)}, 1 AS disposition, 'k2' AS db
  FROM k2pandc
  WHERE disposition = 'CONFIRMED' AND default_flag = 1
)"""

query_toi = rf"""
SELECT {', '.join(cols_toi)}, 0 AS disposition, 'toi' AS db
FROM toi
WHERE tfopwg_disp = 'PC' OR tfopwg_disp = 'APC'
UNION (
  SELECT {', '.join(cols_toi)}, 1 AS disposition, 'toi' AS db
  FROM toi
  WHERE tfopwg_disp = 'CP' OR tfopwg_disp = 'KP'
)"""



#------------------------------------------------------------------------------
# UTILITY FUNCTIONS
#------------------------------------------------------------------------------

def build_simbad_query(table_name: str, *params: list[str]|str, notnull: list[str]|str=None):
    if isinstance(params, list):
        params = ",".join(params)

    s = simbad_query_template.format(params, table_name, table_name)

    if isinstance(notnull, str):
        notnull = [notnull]
    if isinstance(notnull, list):
        s_notnull = "WHERE " + " ".join([f"{_s} IS NOT NULL" for _s in notnull])
        s += s_notnull

    s += ";"

    return s



def form_query(t_name, p_names, notnull: list[str]|str=None):
    cols = ["tbl.oidref"]
    if isinstance(p_names, list):
        cols.extend(p_names)
    else:
        cols.append(p_names)

    cols = ", ".join(cols)

    qs = simbad_query_template.format(cols, t_name)

    if isinstance(notnull, str):
        notnull = [notnull]
    if isinstance(notnull, list):
        s_notnull = "WHERE " + " ".join([f"{_s} IS NOT NULL" for _s in notnull])
        qs += s_notnull

    qs += ";"

    return qs


def submit_query(qs: str, hosts: Table, dtype=np.float64) -> pd.DataFrame:
    print(qs)
    res_vo = service_SIMBAD.run_sync(qs, maxrec=200_000, uploads={"hosts": hosts})
    res_table = res_vo.to_table()
    res_pd = res_table.to_pandas(index="oidref").astype(dtype)

    return res_pd



def median_low(vals):
    vals = np.ravel(vals)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

        if len(vals) % 2 == 0:
            med = np.nanmedian(vals[:-1])
        else:
            med = np.nanmedian(vals)
    return med



def conform_sptype(s_sptype: str):
    p_sptype = re.compile(r"^\s*([OBAFGKM])\s*(\d(\.\d)?)?\s*(I?V|I{1,3})?")
    m_sptype = p_sptype.match(s_sptype)

    s_sptype_new = ""

    if m_sptype is not None:

        s_sptype_new += m_sptype.group(1)

        if m_sptype.group(2) is not None:
            s_sptype_new += m_sptype.group(2)

        if m_sptype.group(4) is not None:
            s_sptype_new += m_sptype.group(4)

    return s_sptype_new



spt_map = {spt_: 10 * i_ for i_, spt_ in enumerate("OBAFGKM")}
lc_map = {lc_: i_ + 1 for i_, lc_ in enumerate(["I", "II", "III", "IV", "V"])}

def extract_sptype(s_sptype: str, default_lc=None):
    p_sptype = re.compile(r"^\s*([OBAFGKM])\s*(\d(\.\d)?)?\s*(I?V|I{1,3})?")
    m_sptype = p_sptype.match(s_sptype)

    spt = np.nan
    lc = default_lc

    if m_sptype is not None:

        spt = spt_map[m_sptype.group(1)]

        if m_sptype.group(2) is not None:
            spt += float(m_sptype.group(2))

        if m_sptype.group(4) is not None:
            lc = lc_map[m_sptype.group(4)]

    return spt, lc




if __name__ == "__main__":

    hosts_df = pd.read_csv("../db/aliases.csv")
    hosts = Table.from_pandas(hosts_df)
    hosts_df.set_index(["oidref"], inplace=True)

    # TODO match uncertainties with measurements
    # get msmt using closest to mean OR smallest unc

    job_info_queue: list[tuple[AsyncTAPJob, np.dtype, str]] = []

    job_spt = service_SIMBAD.submit_job(qs_spt, maxrec=200_000, uploads={"hosts": hosts})
    job_info_queue.append((job_spt, np.str_, "st_spectype"))

    qs_teff = form_query("mesfe_h", "teff AS st_teff", notnull="teff")
    job_teff = service_SIMBAD.submit_job(qs_teff, maxrec=200_000, uploads={"hosts": hosts})
    job_info_queue.append((job_teff, np.int32, "st_teff"))

    qs_met = form_query("mesfe_h", "fe_h AS st_met", notnull="fe_h")
    job_met = service_SIMBAD.submit_job(qs_met, maxrec=200_000, uploads={"hosts": hosts})
    job_info_queue.append((job_met, np.float32, "st_met"))

    job_prot = service_SIMBAD.submit_job(qs_prot, maxrec=200_000, uploads={"hosts": hosts})
    job_info_queue.append((job_prot, np.float64, "st_rotp"))

    qs_flux = form_query("flux", "flux AS st_flux", notnull="flux")
    job_flux = service_SIMBAD.submit_job(qs_flux, maxrec=200_000, uploads={"hosts": hosts})
    job_info_queue.append((job_flux, np.float64, "st_flux"))

    qs_vmag = form_query("allfluxes", "V AS sy_vmag", notnull="V")
    job_vmag = service_SIMBAD.submit_job(qs_vmag, maxrec=200_000, uploads={"hosts": hosts})
    job_info_queue.append((job_vmag, np.float64, "sy_vmag"))

    qs_kmag = form_query("allfluxes", "K AS sy_kmag", notnull="K")
    job_kmag = service_SIMBAD.submit_job(qs_kmag, maxrec=200_000, uploads={"hosts": hosts})
    job_info_queue.append((job_kmag, np.float64, "sy_kmag"))

    qs_gmag = form_query("allfluxes", "g_ AS sy_gmag", notnull="g_")
    job_gmag = service_SIMBAD.submit_job(qs_gmag, maxrec=200_000,
                                         uploads={"hosts": hosts})
    job_info_queue.append((job_gmag, np.float64, "sy_gmag"))

    qs_rmag = form_query("allfluxes", "r_ AS sy_rmag", notnull="r_")
    job_rmag = service_SIMBAD.submit_job(qs_rmag, maxrec=200_000,
                                         uploads={"hosts": hosts})
    job_info_queue.append((job_rmag, np.float64, "sy_rmag"))

    qs_nuvmag = form_query("allfluxes", "F200W AS sy_nuvmag", notnull="F200W")
    job_nuvmag = service_SIMBAD.submit_job(qs_nuvmag, maxrec=200_000, uploads={"hosts": hosts})
    job_info_queue.append((job_nuvmag, np.float64, "sy_nuvmag"))

    job_dist_pc = service_SIMBAD.submit_job(qs_dist_pc, maxrec=200_000, uploads={"hosts": hosts})
    job_info_queue.append((job_dist_pc, np.float64, "sy_dist_pc"))

    job_dist_kpc = service_SIMBAD.submit_job(qs_dist_kpc, maxrec=200_000, uploads={"hosts": hosts})
    job_info_queue.append((job_dist_kpc, np.float64, "sy_dist_kpc"))

    job_dist_mpc = service_SIMBAD.submit_job(qs_dist_mpc, maxrec=200_000, uploads={"hosts": hosts})
    job_info_queue.append((job_dist_mpc, np.float64, "sy_dist_mpc"))

    job_rad = service_SIMBAD.submit_job(qs_rad, maxrec=200_000, uploads={"hosts": hosts})
    job_info_queue.append((job_rad, np.float64, "st_rad"))

    qs_vsini = form_query("mesrot", "vsini AS st_vsin", notnull="vsini")
    job_vsini = service_SIMBAD.submit_job(qs_vsini, maxrec=200_000, uploads={"hosts": hosts})
    job_info_queue.append((job_vsini, np.float32, "st_vsin"))

    # combine column data as the results are completed
    df = hosts_df.copy()
    df_dist = hosts_df.copy()

    # wait for jobs to finish
    while len(job_info_queue) > 0:
        for job_info in job_info_queue:
            job, dtype, col = job_info
            if job.phase == 'PENDING':
                job.run()

            elif job.phase == 'COMPLETED':
                res = job.fetch_result()
                param: pd.DataFrame = res.to_table().to_pandas(index="oidref")

                # system distance column gets special treatment
                if "sy_dist_" in col:
                    param = param[col].astype(dtype)
                    df_dist = pd.merge(df_dist, param, on="oidref", how="outer")

                else:
                    if dtype == np.float64 or dtype == np.float32:
                        param = param.groupby("oidref")[col].apply(median_low)

                    elif dtype == np.int64 or dtype == np.int32:
                        param = param.fillna(-1)
                        param = param.groupby("oidref")[col].apply(median_low)

                    elif col == 'st_spectype':
                        param = param[col].apply(conform_sptype)

                    param = param.astype(dtype)
                    df = pd.merge(df, param, on="oidref", how="outer")

                print(f"\n{len(param)} records returned for '{col}'.")

                job_info_queue.remove(job_info)
                job.delete()

        sleep(1.0)

    # add distance data to table
    sy_dist = df_dist.groupby("oidref")[["sy_dist_pc", "sy_dist_kpc", "sy_dist_mpc"]].apply(median_low)
    sy_dist = pd.Series(data=sy_dist, name="sy_dist")
    df = pd.merge(df, sy_dist, on="oidref", how="outer")
    df["st_rotpref"] = df["st_rotp"].apply(lambda p: None if pd.isna(p) else "simbad")

    df.reset_index(inplace=True, drop=False)

    filename_simbad = "../db/params-simbad.csv"
    print("\nSaving host parameters from SIMBAD to ", filename_simbad)
    df.to_csv(filename_simbad, index=False)



    aliases = pd.read_csv("../db/aliases.csv")

    res_pscomppars = service_NEA.run_sync(query_pscomppars)
    df_pscomppars = res_pscomppars.to_table().to_pandas()
    df_pscomppars["st_rotpref"] = df_pscomppars["st_rotp"].apply(lambda p: None if pd.isna(p) else "pscp")
    df_pscomppars = pd.merge(df_pscomppars, aliases, on="hostname")
    df_nea = df_pscomppars

    res_cumulative = service_NEA.run_sync(query_cumulative)
    df_cumulative = res_cumulative.to_table().to_pandas()
    df_cumulative = pd.merge(df_cumulative, aliases, left_on="kepid", right_on="kic")
    df_cumulative["pl_letter"] = df_cumulative["pl_name"].apply(lambda s: s[-1] if s is not None and len(s) > 0 else "")
    df_nea = concat_update_df(df_nea, df_cumulative)

    res_k2pandc = service_NEA.run_sync(query_k2pandc)
    df_k2pandc = res_k2pandc.to_table().to_pandas()
    df_k2pandc["st_rotpref"] = df_k2pandc["st_rotp"].apply(lambda p: None if pd.isna(p) else "k2pandc")
    df_k2pandc = pd.merge(df_k2pandc, aliases, on="hostname")
    df_nea = concat_update_df(df_nea, df_k2pandc)

    res_toi = service_NEA.run_sync(query_toi)
    df_toi = res_toi.to_table().to_pandas()
    df_toi = pd.merge(df_toi, aliases, left_on="tid", right_on="tic")
    df_nea = concat_update_df(df_nea, df_toi)

    # remove duplicate planet names -- default to PSCP > KOI > K2 > TOI
    df_nea = df_nea.drop_duplicates(subset=["oidref", "pl_letter"], keep="first")
    df_nea.reset_index(drop=True, inplace=True)


    # export data only from NEA
    df_nea.to_csv("../db/params-nea.csv", index=False)

    # okay the merge thing is buggy wtf
    df = pd.read_csv("../db/params-simbad.csv")
    df_nea = pd.read_csv("../db/params-nea.csv")

    # merge by updating rows
    df_all = merge_update_df(df_nea, df)
    # df_all = merge_update_df(df_all, df) # spectype thing still not fixed

    df_all["st_spectype"] = df_all["st_spectype"].fillna("")
    extracted_spt_lc = df_all["st_spectype"].apply(extract_sptype)
    df_all["st_spt_num"] = extracted_spt_lc.apply(lambda t: t[0])
    df_all["st_lc_num"] = extracted_spt_lc.apply(lambda t: t[1])

    # export merged NEA and SIMBAD data
    filename_merged = "../db/params.csv"
    print("Saving host parameters merged from SIMBAD and NEA to ", filename_merged)
    print(df_all.columns)
    df_all.to_csv(filename_merged, index=False)

