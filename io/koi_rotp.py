from astropy.io import ascii
import numpy as np
import pandas as pd

# aliases


# NEA Kepler
table_nea_koi = pd.read_csv("../db/nea/koi.csv")
table_nea_koi = table_nea_koi.rename({"kepid": "KIC"}, axis=1)
table_nea_koi["koi_eccen"] = table_nea_koi["koi_eccen"].fillna(0.0)

# Santos+ 19
table_s19 = ascii.read("../db/kepler/santos19.mrt").to_pandas()
table_s19 = table_s19[["KIC", "Prot", "E_Prot"]]
table_s19["db"] = "s19"
table_s19_koi = pd.merge(table_s19, table_nea_koi, on="KIC", how="inner")

# Santos+ 21
table_s21 = ascii.read("../db/kepler/santos21.mrt").to_pandas()
table_s21 = table_s21[["KIC", "Prot", "E_Prot"]]
table_s21["db"] = "s21"
table_s21_koi = pd.merge(table_s21, table_nea_koi, on="KIC", how="inner")

# Reinhold+ 23
table_r23 = ascii.read("../db/kepler/reinhold23.mrt", data_start=2).to_pandas()
table_r23 = table_r23[["KIC", "ProtFin"]]
table_r23 = table_r23.rename({"ProtFin": "Prot"}, axis=1)
table_r23["db"] = "r23"
table_r23_koi = pd.merge(table_r23, table_nea_koi, on="KIC", how="inner")

# Schochet+ 25
table_s25 = pd.read_csv("../db/asas-sn/schochet25.csv")
table_s25 = table_s25[["KIC", "period", "sigma"]]
table_s25 = table_s25.rename({"period": "Prot", "sigma": "E_Prot"}, axis=1)
table_s25["db"] = "s25"
table_s25_koi = pd.merge(table_s25, table_nea_koi, on="KIC", how="inner")

table_koi = pd.concat([table_s19_koi,
                       table_s21_koi,
                       table_r23_koi,
                       table_s25_koi],
                      ignore_index=True)
print(table_koi)
print(table_koi["KIC"].nunique())
print(table_koi.count())

table_koi.to_csv("../db/kepler/koi_master.csv")
