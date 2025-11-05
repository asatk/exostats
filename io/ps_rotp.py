from astropy.io import ascii
import pandas as pd

# PSCP table
ps = pd.read_csv("../db/nea/ps.csv")

# aliases (KOI, KIC, TOI, KIC)
names = pd.read_csv("./aliases.csv")


