import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from typing import List
import matplotlib.pyplot as plt


def ARIMAd(z:pd.DataFrame, alpha:float):
    """This function determines the differencing order parameter for an ARIMA model. It will perform an
     Augemented Dikcey Fuller test to determine the minimum order d at which the null hypothesis
     (that the series is not stationary) is rejected. It takes as input a pandas DataFrame containing the time series
     and an alpha value for the hypothesis test. Currently, it will only test up to d=5. It returns the differencing
      order d (int) """
    nullRejected = False
    d = 0
    while not nullRejected and d<6:
        pvalue = adfuller(z)[1]
        if pvalue < alpha:
            # Null hypothesis is rejected, series is likely staionary
            return d
        else:
            # Need to difference again and track d
            d+=1
            z = z.diff().dropna()
            if d==6:
                print('Could not find d<6')
                return None
def ARIMApq(z:pd.DataFrame,d:int,p_space:List[int],q_space:List[int]):
    """this function handles determining p and q parameters for an ARIMA model of known differencing order.
     It takes as input a pandas DataFrame containing the time series and the parameter spaces for p and q to sweep over
     which should be lists of integers. The function will perform a parameter sweep and find the combination that
      minimises Bayesian Information Criterion. It turns a tuple of the model order parameters (p,d,q)."""
    bic = np.inf
    order = None
    print('pspace', p_space)
    print('qspace', q_space)
    for p in p_space:
        for q in q_space:
            try:
                model = ARIMA(z,order=(p,d,q))
                fit = model.fit(method_kwargs={"warn_convergence": False})
                if fit.bic < bic:
                    order = (p,d,q)
                    bic = fit.bic
            except Exception as e:
                continue
    print('Found ', order)
    return order

def getARIMAParams(z:pd.DataFrame,d_alpha:float,p_space:List[int],q_space:List[int]):
    """This function handles determining the ARIMA parameters for a data frame of time series. It takes as input a
    DataFrame of time series, the critical value for the Augmented Dickey Fuller hypothesis test to determine ARIMA
     differencing order d and parameter spaces for p and q over which to sweep. It returns a DataFrame indexed by model
      order containing the ARIMA parameters for each time series"""
    results = pd.DataFrame(index = ['p','d','q'], columns=z.columns)
    for c in z.columns:
        d = ARIMAd(z[c], alpha=d_alpha)
        print('For {} found d = {}'.format(c, d))
        order = ARIMApq(z[c],d,p_space,q_space)
        results[c] = order
    return results

def getPredictions(z:pd.DataFrame, orders):
     pred_df = pd.DataFrame(index = z.index, columns=z.columns)
     for c in z.columns:
         model_params = orders[c]
         model = ARIMA(z[c], order=model_params)
         fit = model.fit()
         predictions = fit.predict(start=0, end=len(z)-1, exog=None, dynamic=False)
         pred_df[c] = predictions
         plt.figure()
         pred_df[c].plot(label='{} Predictions'.format(c))
         z[c].plot(label='{} Observed'.format(c))
         plt.legend()
         plt.show()
     print(pred_df.head())