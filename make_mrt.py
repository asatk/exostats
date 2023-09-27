from astropy.io import ascii
# import numpy as np
# import pandas as pd


if __name__ == "__main__":
    fname = "current-exo-data/alfven_data.csv"
    table = ascii.read(fname, format="csv")
    # (fname, format="csv")

    ascii.write(table, "paper_table.mrt", overwrite=True, format="mrt")

    # # ascii.AASTex()
    # mrt = ascii.Mrt()
    # mrt_text = mrt.write(table)
    # print(mrt_text)