import abc
import numpy as np
from numpy import linalg as la
import pandas as pd
from typing import Literal, List


def _loady(y, df):
    # Get response vector y from DataFrame
    if isinstance(y, str):
        if (df is None):
            print(f"No DataFrame provided for predictor column {y}")
            return None
        y = df[y].to_numpy()
    return y

def _loadx(x, df, cat):
    # Get data matrix x from columns of DataFrame
    # if isinstance(x, (str, list[str])):
    if isinstance(x, (str, list)):
        if isinstance(x, str):
            x = [x]
        if (df is None):
            print(f"No DataFrame provided for regressor columns {x}")
            return None
        
        # TODO make sure all in cat are in x
        if cat is not None:
            # Convert categorical variable to integers 0:k
            for cvar in cat:
                cvec = np.unique(df[cvar], return_inverse=True)[1]
                df.loc[cvar] = cvec

        x = np.insert(df[x].to_numpy(), 0, axis=1, values=1.0)
    return x




class LRFit(abc.ABCMeta):

    @abc.abstractmethod
    def learn(y: np.ndarray|str,
              x: np.ndarray|str|list[str],
              df: pd.DataFrame=None,
              cat: str|list[str]=None,
              *args):
        ...

    @abc.abstractmethod
    def predict(d: dict[str, np.ndarray],
                x: np.ndarray|str|list[str],
                df: pd.DataFrame=None,
                cat: str|list[str]=None):
        ...
        
        




def _lasso(y: np.ndarray,
           x: np.ndarray,
           lam: np.float64=None,
           eta: np.float64=1.0,
           tol: np.float64=1.e-10):
    """
    Project GD?
    """

    if lam is None:
        
        lam_prev = 0.0
        lam_curr = 1.0
        
        while (lam_curr - lam_prev > tol):
            
            diff_vec = np.abs(y) - lam_curr
            flam = np.sum(np.clip(diff_vec, a_min=0.0, a_max=None)) - 1
            dflam = -1 * np.sum(diff_vec > 0)
            
            lam_prev = lam_curr.copy()
            lam_curr = lam_curr - flam / dflam

        lam = lam_curr

    b_prev = 0.0
    b_curr = 1.0

    while(b_curr - b_prev > tol):

        b_prev = b_curr
        b_temp = b_curr - eta * x.T @ (y - x @ b_curr)
        b_curr = np.sign(b_temp) * np.clip(np.abs(b_temp) - lam, a_min=0.0, a_max=None)
    
    return b_curr

def learnLM(y: np.ndarray|str,
            x: np.ndarray|str|list[str],
            df: pd.DataFrame=None,
            cat: str|list[str]=None,
            kind: Literal["ols, ridge, lasso"]="ols",
            *args) -> dict[str, np.ndarray]|None:
    
    # Get response vector y from DataFrame
    if isinstance(y, str):
        if (df is None):
            print(f"No DataFrame provided for predictor column {y}")
            return None
        y = df[y].to_numpy()
    
    # Get data matrix x from columns of DataFrame
    # if isinstance(x, (str, list[str])):
    if isinstance(x, (str, list)):
        if isinstance(x, str):
            x = [x]
        if (df is None):
            print(f"No DataFrame provided for regressor columns {x}")
            return None
        
        # TODO make sure all in cat are in x
        if cat is not None:
            # Convert categorical variable to integers 0:k
            for cvar in cat:
                cvec = np.unique(df[cvar], return_inverse=True)[1]
                df.loc[cvar] = cvec

        x = np.insert(df[x].to_numpy(), 0, axis=1, values=1.0)

    # Expand categorical variables based on unique values in the column
    # np.insert(axis=1)
    
    hat = None
    # Calculate regression parameters
    if kind == "ols":

        if x.shape[0] < x.shape[1]:
            print(f"Problem is underdetermined (n={x.shape[0]} < {x.shape[1]} = p)")
            return None

        cov = la.inv(x.T @ x)
        covx = cov @ x.T
        hat = x @ covx
        b = covx @ y
        res = y - x @ b
    elif kind == "ridge":
        if len(args) < 1:
            print("No argument provided for Ridge Regression weight")
            return None
        
        lam = np.abs(args[0])
        mtx = la.inv(x.T @ x - lam * np.identity(x.shape[1])) @ x.T
        b = mtx @ y
        res = y - x @ b

    elif kind == "lasso":
        if len(args) < 1:
            print("No argument provided for Ridge Regression weight")
            return None
        lam = np.abs(args[0])
        b = _lasso(y, x, lam)
        res = y - x @ b
    else:
        print(f"Kind {kind} is not implemented.")
        return None

    d = {"coefs": b,
         "hat": hat,
         "residuals": res,
         "kind": kind}

    return d


def predictLM(d: dict[str, np.ndarray],
              x: np.ndarray|str|list[str],
              df: pd.DataFrame=None,
              cat: str|list[str]=None):
    
    b = d.get("coefs", None)
    if b is None:
        print("No regression coefficients provided in model.")
        return None
    
    # Get new data from columns of DataFrame
    # if isinstance(x, (str, list[str])):
    if isinstance(x, (str, list)):
        if (df is None):
            print(f"No DataFrame provided for regressor columns {x}")
            return None
        x = np.insert(df[x].to_numpy(), 0, axis=1, values=1.0)
    if (x.shape[1] != b.shape[0]):
        print(f"Number of data paramters ({x.shape[1]}) does not match number of model coefficients ({b.shape[0]})")

    yhat = x @ b

    return yhat


# def pruneLM(d: dict[str, np.ndarray],
#             x: np.ndarray,
#             y: np.ndarray)



    

if __name__ == "__main__":
    print("No main functionality")