import pandas as pd
import pmdarima
from pmdarima.arima import ndiffs
from typing import List
import warnings
from joblib import Parallel, delayed
import exog

warnings.filterwarnings('ignore', 'Non-stationary starting autoregressive parameters found. Using zeros '
                                  'as starting parameters.')
warnings.filterwarnings('ignore', 'Non-invertible starting MA parameters found. Using zeros as starting '
                                  'parameters.')
warnings.filterwarnings('ignore', category=FutureWarning)



def ARIMAd(z: pd.DataFrame, alpha: float, test:str='adf', max_d=6)->int:
    """ Handles determining the differencing order parameter (d) for ARIMA at the desired confidence level, alpha."""
    d = pmdarima.arima.ndiffs(z, alpha=alpha, test=test, max_d=max_d)
    return d

def train_single_ARIMA_model(column, z_train, d_alpha,L:int):
    """ Single model trainer for parallel processing """
    print(f'Finding ARIMA parameters for {column}')
    d = ARIMAd(z_train[column], alpha=d_alpha)
    ex = exog.getAllignedExogLags(z_train[column],L)
    model = pmdarima.auto_arima(z_train[column],d=d, seasonal=False, trace=False)
    return (column, model)

def trainARIMA(z: pd.DataFrame, train_window: int,L:int, d_alpha=0.05,parallel=True)->dict:
    ''' Handles model training. Parallelised by default. Returns stock ticker keyed dict of models.'''
    z_train = z.iloc[:train_window, :]
    print('Training ARIMA models')

    if parallel:
        # Run ARIMA training in parallel for each column
        models = Parallel(n_jobs=-1)(delayed(train_single_ARIMA_model)(col, z_train, d_alpha,L) for col in z_train.columns)

        # Convert list of results to dictionary of models
        models_dict = {column: model for column, model in models}
        return models_dict
    else:
        models = {c: None for c in z_train.columns}
        for c in z_train.columns:
            print('Finding ARIMA parameters for ', c)
            d = ARIMAd(z_train[c], alpha=d_alpha)
            ex = exog.getAllignedExogLags(z_train[c],L)
            models[c] = pmdarima.auto_arima(z_train[c],exog=ex, d=d, seasonal=False, trace=True,error_action='ignore')
        return models


