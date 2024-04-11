from stats import OLSFit
from stats.utils import load

if __name__ == "__main__":
    
    load(x=(),
         y="",
         df="tables/nasa_exo_PSCP.csv",
         cat=None)

    ols = OLSFit()