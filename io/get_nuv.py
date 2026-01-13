from astropy.io import ascii
import pandas as pd
import numpy as np

aliases = ascii.read("../db/aliases.csv", format="csv").to_pandas()

prot = ascii.read("../db/prot.csv", format="csv").to_pandas()

galex = ascii.read("../db/olmedo-galex.mrt").to_pandas()

galex = galex.loc[galex["KIC"].notnull(), ["KIC", "NUVmag", "e_NUVmag"]]
galex = galex.rename({"KIC": "kic"}, axis=1)

nuv_kic = pd.merge(prot, galex, on="kic", how="inner")
u = np.unique(nuv_kic["oidref"], return_index=True)[1]
print(u)
print(nuv_kic.loc[u].count())

