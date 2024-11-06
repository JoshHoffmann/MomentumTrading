import pandas as pd
import numpy as np
import pmdarima
from pmdarima.arima import ndiffs
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from typing import List
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore', 'Non-stationary starting autoregressive parameters found. Using zeros '
                                  'as starting parameters.')
warnings.filterwarnings('ignore', 'Non-invertible starting MA parameters found. Using zeros as starting '
                                  'parameters.')
warnings.filterwarnings('ignore', category=FutureWarning)

def ARIMAd(z: pd.DataFrame, alpha: float,test='adf',max_d=6):
    """This function determines the differencing order parameter for an ARIMA model. It will perform an
     Augmented Dickey Fuller test to determine the minimum order d at which the null hypothesis
     (that the series is not stationary) is rejected. It takes as input a pandas DataFrame containing the time series
     and an alpha value for the hypothesis test. Currently, it will only test up to d=5. It returns the differencing
      order d (int) """
    d = ndiffs(z, alpha=alpha, test=test, max_d=max_d)
    return d


def ARIMApq(z: pd.DataFrame, d: int, p_space: List[int], q_space: List[int]):
    """this function handles determining p and q parameters for an ARIMA model of known differencing order.
     It takes as input a pandas DataFrame containing the time series and the parameter spaces for p and q to sweep over
     which should be lists of integers. The function will perform a parameter sweep and find the combination that
      minimises Bayesian Information Criterion. It turns a tuple of the model order parameters (p,d,q)."""

    order = pmdarima.auto_arima(z,d=d,seasonal=False, stepwise=True,suppress_warnings=True, error_action="ignore",
                                max_p=6,max_order=None, trace=True).order
    return order


def getARIMAParams(z: pd.DataFrame, d_alpha: float, p_space: List[int], q_space: List[int]):
    """This function handles determining the ARIMA parameters for a data frame of time series. It takes as input a
    DataFrame of time series, the critical value for the Augmented Dickey Fuller hypothesis test to determine ARIMA
     differencing order d and parameter spaces for p and q over which to sweep. It returns a DataFrame indexed by model
      order containing the ARIMA parameters for each time series"""
    results = pd.DataFrame(index=['p', 'd', 'q'], columns=z.columns)
    for c in z.columns:
        print('Getting ARIMA parameter d for ', c)
        d = ARIMAd(z[c], alpha=d_alpha)
        print('Found d = ', d)
        print('Getting ARIMA parameters p & q for ', c)
        order = ARIMApq(z[c], d, p_space, q_space)
        print('Found order = ', order)
        results[c] = order
    return results

def trainARIMA(z:pd.DataFrame,train_window:int,d_alpha=0.05):
    z_train=z.iloc[:train_window,:]
    print('TRAIN')
    print(len(z_train))
    models = {c: None for c in z_train.columns}
    for c in z_train.columns:
        print('Finding ARIMA parameters for ', c)
        d = ARIMAd(z_train[c], alpha=d_alpha)
        models[c] = pmdarima.auto_arima(z_train[c],d=d, seasonal=False, trace=True)
    return models








