import abc
import numpy as np
from numpy import linalg as la
import pandas as pd
from typing import Literal, List

from .utils import load


class LRFit(metaclass=abc.ABCMeta):

    def __init__(self):
        self._learned: bool=False
        self.coefs: np.ndarray=None
        self.res: np.ndarray=None

    @abc.abstractmethod
    def learn(self,
              y: np.ndarray,
              x: np.ndarray,
              *args):
        ...

    def predict(self,
                x: np.ndarray):
        
        if not self._learned:
            print("The data have not been regressed on yet.")
            return None
        
        return x @ self.coefs


class OLSFit(LRFit):

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray):
        super().__init__()
        self.learn(x, y)


    def learn(self,
              x: np.ndarray,
              y: np.ndarray,
              *args):

        if x.shape[0] < x.shape[1]:
            print(f"Problem is underdetermined (n = {x.shape[0]} < {x.shape[1]} = p)")
            return None

        cov = la.inv(x.T @ x)
        covx = cov @ x.T
        hat = x @ covx
        b = covx @ y
        res = y - x @ b
        s_hat = np.sum(np.square(res)) / (x.shape[0] - y.shape[0] - 2)

        self.coefs = b
        self._cov = cov
        self._hat: np.ndarray = hat
        self.res: np.ndarray = res
        self._s_hat = s_hat

        self._learned = True

    def outliers(self,
                 type: Literal["leverage", "quantile", "stdres"]="leverage",
                 tol: float=None):
        """
        Uses Sheather's heuristics for outlier hii > 2*p+1/n & abs(rhat) > 2
        tol is IQR factor or sigma
        """
        hii = np.diag(self._hat)
        lev = hii > 2 * (self.coefs.shape[0] + 1) / self.res.shape[0]

        if type == "quantile":
            if tol is None:
                tol = 1.5
            q = np.percentile(self.res, [0.25, 0.75])
            q1 = q[0]
            q3 = q[1]
            iqr = q3 - q1
            bad = (self.res < (q1 - tol * iqr)) | (self.res > (q3 + tol * iqr))
        elif type == "stdres":
            if tol is None:
                tol = 2
            bad = np.square(self.res) > tol * self._s_hat * (1 - np.diag(self._hat))
        else:
            bad = 1
        
        out = lev * bad
        return np.nonzero(out)


class RidgeFit(LRFit):

    def _standardize(self, x: np.ndarray):
        return x / np.sqrt(np.sum(np.square(x - np.mean(axis=0))) / x.shape[0])

    def learn(self,
              y: np.ndarray,
              x: np.ndarray,
              *args):
        
        self._standardize(x)


class LassoFit(LRFit):

    def learn(self,
              y: np.ndarray,
              x: np.ndarray,
              *args):
        """
        Project GD?
        """

        lam = args[0] #None
        eta = args[1] #1.0
        tol = args[2] #1e-10

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


if __name__ == "__main__":
    print("No main functionality")