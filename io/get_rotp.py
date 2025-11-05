from astropy.io import ascii
import numpy as np
import pandas as pd


ps = pd.read_csv("../db/nea/ps.csv")
k2 = pd.read_csv("../db/nea/k2.csv")
koi = pd.read_csv("../db/nea/koi.csv")
toi = pd.read_csv("../db/nea/toi.csv")

ps_hosts = ps["hostname"]
k2_hosts = k2["hostname"]
koi_hosts = koi["kepid"].apply(lambda num: f"KIC {num:d}")
toi_hosts = toi["tid"]

hostnames = .






def collect_by_tag(tag: str):
    ...

collect_by_tag("TIC")


#------------------------------------------------------------------------------
# NEA DATABASES
#------------------------------------------------------------------------------

# NEA PSCP
nea_ps = pd.read_csv("../db/nea/ps.csv")

# PSCP aliases (KOI, KIC, EPIC, TOI, TIC)
aliases = pd.read_csv("../db/nea/aliases.csv")

# NEA Kepler
nea_koi = pd.read_csv("../db/nea/koi.csv")
nea_koi = nea_koi.rename({"kepid": "KIC"}, axis=1)
nea_koi["koi_eccen"] = nea_koi["koi_eccen"].fillna(0.0)

# NEA K2
nea_k2 = pd.read_csv("../db/nea/k2.csv")

# NEA TOI
nea_toi = pd.read_csv("../db/nea/toi.csv")
nea_toi = nea_toi.rename({"tid": "TIC", "toipfx": "TOI"}, axis=1)









#------------------------------------------------------------------------------
# ASAS-SN DATABASES
#------------------------------------------------------------------------------

# Schochet+ 25
asassn_s25 = pd.read_csv("../db/asas-sn/schochet25.csv")
asassn_s25 = asassn_s25[["dr3_source_id", "dr2_source_id", "KIC", "tic_id", "period", "sigma"]]
asassn_s25 = asassn_s25.rename({"period": "Prot",
                                "sigma": "E_Prot",
                                "tic_id": "TIC",
                                "dr2_source_id": "gaia_dr2_id",
                                "dr3_source_id": "gaia_dr3_id"}, axis=1)
asassn_s25["db"] = "s25"



#------------------------------------------------------------------------------
# KEPLER DATABASES
#------------------------------------------------------------------------------

# Santos+ 19
kep_s19 = ascii.read("../db/kepler/santos19.mrt").to_pandas()
kep_s19 = kep_s19[["KIC", "Prot", "E_Prot"]]
kep_s19["db"] = "s19"

# Santos+ 21
kep_s21 = ascii.read("../db/kepler/santos21.mrt").to_pandas()
kep_s21 = kep_s21[["KIC", "Prot", "E_Prot"]]
kep_s21["db"] = "s21"

# Reinhold+ 23
kep_r23 = ascii.read("../db/kepler/reinhold23.mrt", data_start=2).to_pandas()
kep_r23 = kep_r23[["KIC", "ProtFin", "SNR"]]
kep_r23 = kep_r23.rename({"ProtFin": "Prot"}, axis=1)
kep_r23["E_Prot"] = kep_r23["Prot"] / kep_r23["SNR"]
kep_r23["db"] = "r23"
kep_r23 = kep_r23.drop(columns="SNR")



#------------------------------------------------------------------------------
# TESS DATABASES
#------------------------------------------------------------------------------

# Colman+ 24
table_c24 = pd.read_csv("../db/tess/colman24.csv")
table_c24 = table_c24[["TIC", "Period", "Period sigma"]]
table_c24 = table_c24.rename({"Period": "Prot", "Period sigma": "E_Prot"}, axis=1)
table_c24["db"] = "c24"



#------------------------------------------------------------------------------
# MASTER DATABASES
#------------------------------------------------------------------------------

# NEA confirmed planets

# GAIA DR3 from s25
#ps_s25_gdr2 = pd.merge(nea_ps, asassn_s25[["Prot", "E_Prot", "gaia_dr2_id"]], on="gaia_dr2_id", how="inner")
#ps_s25_gdr3 = pd.merge(nea_ps, asassn_s25[["Prot", "E_Prot", "gaia_dr3_id"]], on="gaia_dr3_id", how="inner")
#ps_s25_kic = pd.merge(nea_ps, asassn_s25[["Prot", "E_Prot", "KIC"]], on="KIC", how="inner")
#ps_s25_tic = pd.merge(nea_ps, asassn_s25[["Prot", "E_Prot", "TIC"]], on="TIC", how="inner")

#ps_s25 = pd.concat([ps_s25_gdr2, ps_s25_gdr3, ps_s25_kic, ps_s25_tic])
#ps_s25 = pd.merge(nea_ps, asassn_s25, on=["gaia_dr2_id", "gaia_dr3_id", "KIC", "TIC"], how="inner")
ps_s25 = pd.merge(nea_ps, asassn_s25, on=["gaia_dr2_id"], how="inner")
ps_s25.to_csv("../db/master/ps_s25.csv")

pd.set_option("display.max_rows", 200)
print(ps_s25.sort_values(by="pl_name"))


"""

# Kepler planets and candidates
table_s19_koi = pd.merge(kep_s19, nea_koi, on="KIC", how="inner")
table_s21_koi = pd.merge(kep_s21, nea_koi, on="KIC", how="inner")
table_r23_koi = pd.merge(kep_r23, nea_koi, on="KIC", how="inner")
asassn_s25_koi = pd.merge(asassn_s25, nea_koi, on="KIC", how="inner")

table_koi = pd.concat([table_s19_koi,
                       table_s21_koi,
                       table_r23_koi,
                       asassn_s25_koi],
                      ignore_index=True)

table_koi.to_csv("../db/master/koi.csv")

# K2 planets and candidates


# TESS
table_c24_toi = pd.merge(table_c24, table_nea_toi, on="TIC", how="inner")
asassn_s25_toi = pd.merge(asassn_s25, table_nea_toi, on="TIC", how="inner")

table_toi = pd.concat([table_c24_toi,
                       asassn_s25_toi],
                      ignore_index=True)

table_toi.to_csv("../db/master/toi.csv")
"""
