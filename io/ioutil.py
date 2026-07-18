import pandas as pd

def concat_update_row(df: pd.DataFrame) -> pd.DataFrame:
    row_default = df.iloc[0].copy()

    if len(df) == 1:
        return row_default

    row_new = df.iloc[1]

    for icol, (col, oldval) in enumerate(row_default.items()):
        oldval = row_default[col]

        if col not in row_new.index:
            continue

        newval = row_new[col]

        if pd.isna(oldval):
            row_default.at[col] = newval

    return row_default

def concat_update_df(df_default: pd.DataFrame, df_update: pd.DataFrame):
    df_concat = pd.concat([df_default, df_update])
    df_groupby = df_concat.groupby(by="pl_name")
    df = df_groupby.apply(concat_update_row, include_groups=False).reset_index(level=0).reset_index(drop=True)
    return df

def merge_update_row(row: pd.Series) -> pd.Series:
    row = row.copy()
    for col, val in row.items():
        col_simbad = col + "_simbad"
        if col not in row.index or col in row.index and col_simbad not in row.index:
            continue

        if pd.isna(val):
            row[col] = row[col_simbad]

        row = row.drop(col_simbad)

    return row


def merge_update_df(df_default: pd.DataFrame, df_update: pd.DataFrame):
    df_merge = pd.merge(df_default, df_update, how="outer", on="oidref", suffixes=(None, "_simbad"))
    df = df_merge.apply(merge_update_row, axis=1)
    return df