import numpy as np
import pandas as pd
import os

def load(x: str|list[str],
         y: str,
         df: str|pd.DataFrame,
         cat: str|list[str]=None) -> tuple[np.ndarray, np.ndarray]:
    
    if isinstance(df, str):
        if df[0] != "/" and df[0] != "~":
            df = os.getcwd() + "/" + df
        df = pd.read_csv(df)

    # Select rows that have valid records for each column
    logic = 1

    # Get data matrix x from columns of DataFrame
    if isinstance(x, (str, list)):
        if isinstance(x, str):
            x = [x]
        if (df is None):
            print(f"No DataFrame provided for regressor columns {x}")
            return None
        
        logic *= np.prod([df[col].notnull() for col in x], axis=0)
        
        # TODO make sure all in cat are in x
        if cat is not None:
            # Convert categorical variable to integers 0:k
            for cvar in cat:
                cvec = np.unique(df[cvar], return_inverse=True)[1]
                df.loc[cvar] = cvec

        x = np.insert(df[x].to_numpy(), 0, axis=1, values=1.0)

    # Get response vector y from DataFrame
    if isinstance(y, str):
        if (df is None):
            print(f"No DataFrame provided for predictor column {y}")
            return None
        
        logic *= df[y].notnull()

        y = df[y].to_numpy()

    logic = logic == 1

    return (x[logic], y[logic])