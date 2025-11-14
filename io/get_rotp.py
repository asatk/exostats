from astropy.io import ascii
import numpy as np
import pandas as pd


# load table of aliases for all exoplanet detections (candidates and confirmed)
aliases = ascii.read("../db/aliases.csv").to_pandas()


#------------------------------------------------------------------------------
# GAIA DR3 ID HOSTS
#------------------------------------------------------------------------------

# Lu+ 22
lu22 = ascii.read("../db/gaia/lu22.mrt").to_pandas()
lu22 = lu22[["EDR3", "Prot"]]
lu22 = lu22.rename({"EDR3": "gaia_dr3_id"}, axis=1)
lu22["db"] = "lu22"

# Christy+ 23 -- G Band data
chr23g = ascii.read("../db/gaia/christy23_gband.csv", format="csv").to_pandas()
chr23g = chr23g.loc[chr23g["ML_classification"] == "ROT", ["EDR3_source_id", "Period"]]
chr23g = chr23g.rename({
    "EDR3_source_id": "gaia_dr3_id",
    "Period": "Prot"
    }, axis=1)
chr23g["db"] = "chr23g"

# Christy+ 23 -- V Band data
chr23v = ascii.read("../db/gaia/christy23_vband.csv", format="csv").to_pandas()
chr23v = chr23v.loc[chr23v["variable_type"] == "ROT", ["edr3_source_id", "period"]]
chr23v = chr23v.rename({
    "edr3_source_id": "gaia_dr3_id",
    "period": "Prot"
    }, axis=1)
chr23v["db"] = "chr23v"

# Claytor+ 25
cla25 = ascii.read("../db/gaia/claytor25.csv", format="csv").to_pandas()
cla25 = cla25[["gaia_designation", "period", "period_err"]]
cla25 = cla25.rename({
    "period": "Prot",
    "period_err": "E_Prot"
    }, axis=1)
cla25["gaia_dr3_id"] = cla25["gaia_designation"].apply(lambda s: int(s[-19:]))
cla25 = cla25.drop("gaia_designation", axis=1)
cla25["db"] = "cla25"

# Schochet+ 25
sch25 = ascii.read("../db/gaia/schochet25.csv", format="csv").to_pandas()
sch25 = sch25[["dr3_source_id", "period", "sigma"]]
sch25 = sch25.rename({
    "dr3_source_id": "gaia_dr3_id",
    "period": "Prot",
    "sigma": "E_Prot",
    }, axis=1)
sch25["db"] = "sch25"

# concatenate all prot tables using gaia_dr3_id to identify hosts
prot_gdr3 = pd.concat([lu22, chr23g, chr23v, cla25, sch25])
prot_gdr3 = pd.merge(aliases, prot_gdr3, on="gaia_dr3_id", how="left")
print("gaia_dr3_id: " + str(prot_gdr3.loc[prot_gdr3["Prot"].notnull(), "oidref"].nunique()))



#------------------------------------------------------------------------------
# KIC HOSTS
#------------------------------------------------------------------------------

# McQuillan+ 13
mcq13 = ascii.read("../db/kic/mcquillan13.mrt").to_pandas()
mcq13 = mcq13[["KID", "Period", "Period_Err"]]
mcq13 = mcq13.rename({
    "KID": "kic", 
    "Period": "Prot",
    "Period_Err": "E_Prot"
    }, axis=1)
mcq13["db"] = "mcq13"

# McQuillan+ 14
mcq14 = ascii.read("../db/kic/mcquillan14.mrt").to_pandas()
mcq14 = mcq14[["KIC", "PRot", "e_PRot"]]
mcq14 = mcq14.rename({
    "KIC": "kic",
    "PRot": "Prot",
    "e_PRot": "E_Prot"
    }, axis=1)
mcq14["db"] = "mcq14"

# Santos+ 19
san19 = ascii.read("../db/kic/santos19.mrt").to_pandas()
san19 = san19[["KIC", "Prot", "E_Prot"]]
san19 = san19.rename({"KIC": "kic"}, axis=1)
san19["db"] = "san19"

# Santos+ 21
san21 = ascii.read("../db/kic/santos21.mrt").to_pandas()
san21 = san21[["KIC", "Prot", "E_Prot"]]
san21 = san21.rename({"KIC": "kic"}, axis=1)
san21["db"] = "san21"

# Reinhold+ 23
rei23 = ascii.read("../db/kic/reinhold23.mrt").to_pandas()
rei23 = rei23[["KIC", "ProtFin", "SNR"]]
rei23 = rei23.rename({
    "KIC": "kic",
    "ProtFin": "Prot"
    }, axis=1)
rei23["E_Prot"] = rei23["Prot"] / rei23["SNR"]
rei23["db"] = "rei23"
rei23 = rei23.drop(columns="SNR")

# concatenate all prot tables using kic to identify hosts
prot_kic = pd.concat([mcq13, mcq14, san19, san21, rei23])
prot_kic = pd.merge(aliases, prot_kic, on="kic", how="left")
print("kic: " + str(prot_kic.loc[prot_kic["Prot"].notnull(), "oidref"].nunique()))



#------------------------------------------------------------------------------
# EPIC HOSTS
#------------------------------------------------------------------------------

# Saylor+ 18
say18 = ascii.read("../db/epic/saylor18.mrt").to_pandas()
say18 = say18[["EPIC", "Prot", "e_Prot"]]
say18 = say18.rename({
    "EPIC": "epic",
    "e_Prot": "E_Prot"
    }, axis=1)
say18["db"] = "say18"

# concatenate all prot tables using epic to identify hosts
prot_epic = say18
prot_epic = pd.merge(aliases, prot_epic, on="epic", how="left")
print("epic: " + str(prot_epic.loc[prot_epic["Prot"].notnull(), "oidref"].nunique()))



#------------------------------------------------------------------------------
# TIC HOSTS
#------------------------------------------------------------------------------

# Howard+ 20
how20 = ascii.read("../db/tic/howard20.mrt").to_pandas()
how20 = how20[["TIC", "Prot", "e_Prot"]]
how20 = how20.rename({
    "TIC": "tic",
    "e_Prot": "E_Prot"
    }, axis=1)
how20["db"] = "how20"

# Martins+ 20
mar20 = ascii.read("../db/tic/martins20.mrt").to_pandas()
mar20 = mar20[["TIC-ID", "Prot", "e_Prot"]]
mar20 = mar20.rename({
    "TIC-ID": "tic",
    "e_Prot": "E_Prot"
    }, axis=1)
mar20["db"] = "mar20"

# Holcomb+ 22
hol22 = ascii.read("../db/tic/holcomb22.csv", format="csv").to_pandas()
hol22 = hol22.loc[hol22["rotation_accepted"] == "True", ["TICID", "Prot"]]
hol22 = hol22.rename({"TICID": "tic"}, axis=1)
hol22["db"] = "hol22"

# Claytor+ 24
cla24 = ascii.read("../db/tic/claytor24.mrt").to_pandas()
cla24 = cla24[["TIC", "prot", "e_prot"]]
cla24 = cla24.rename({
    "TIC": "tic",
    "prot": "Prot",
    "e_prot": "E_Prot"
    }, axis=1)
cla24["db"] = "cla24"

# Colman+ 24
col24 = ascii.read("../db/tic/colman24.csv", format="csv").to_pandas()
col24 = col24[["TIC", "Period", "Period sigma"]]
col24 = col24.rename({
    "TIC": "tic",
    "Period": "Prot",
    "Period sigma": "E_Prot"
    }, axis=1)
col24["db"] = "col24"

# concatenate all prot tables using tic to identify hosts
prot_tic = pd.concat([how20, mar20, hol22, cla24, col24])
prot_tic = pd.merge(aliases, prot_tic, on="tic", how="left")
print("tic: " + str(prot_tic.loc[prot_tic["Prot"].notnull(), "oidref"].nunique()))


# MASTER PROT TABLE

prot_master = pd.concat([prot_gdr3, prot_kic, prot_epic, prot_tic])
print("all: " + str(prot_master.loc[prot_master["Prot"].notnull(), "oidref"].nunique()))

print(prot_master.count())
print(prot_master[prot_master["Prot"].notnull()].count())

