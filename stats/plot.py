from matplotlib import pyplot as plt
import pandas as pd

def diag_plot(df: pd.DataFrame, out_path: str=None):
    cols = df.columns
    ncols = len(cols)

    fig, ax = plt.subplots(nrows=ncols-1, ncols=ncols-1)


    for i in range(ncols - 1):
        col_main = cols[i]
        for j in range(0, i):
            col_sub = cols[j]
            axij: plt.Axes = ax[i][j]
            axij.scatter(df[col_sub], df[col_main])
            axij.set_xlabel(col_sub)
            axij.set_ylabel(col_main)

    # fig.tight_layout()
    if out_path is not None:
        plt.savefig(out_path)
    plt.show()
