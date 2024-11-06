import pandas as pd
import numpy as np

def Returns(priceData:pd.DataFrame,signal:pd.DataFrame):
    prices = priceData.loc[signal.index, signal.columns]
    returns = (signal.shift(1) * prices.pct_change()).sum(axis=1)
    return returns

def CumulativeReturns(priceData:pd.DataFrame,signal:pd.DataFrame):
    returns = Returns(priceData,signal)
    cumulativeReturns = (1+returns).cumprod()-1
    return cumulativeReturns

def Sharpe(priceData:pd.DataFrame,signal:pd.DataFrame):
    returns = Returns(priceData,signal)
    Sharpe = np.round(returns.mean(axis=0) / returns.std(axis=0), 2)
    return Sharpe