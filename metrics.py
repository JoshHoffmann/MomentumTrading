import pandas as pd
import numpy as np
import backtest

def priceRebalance(priceData,rebalancePeriod):
    rebalancePoints = priceData.resample(rebalancePeriod).last().index

    adj_rebalancePoints = [priceData.index[priceData.index <= point][-1] for point in rebalancePoints]

    w_rebalanced = priceData.loc[adj_rebalancePoints].reindex(priceData.index).ffill().dropna()

    return w_rebalanced

def Returns(priceData:pd.DataFrame,signal:pd.DataFrame,rebalancePeriod:str):

    prices_rebalanced = priceRebalance(priceData,rebalancePeriod).loc[signal.index,:]

    pct = prices_rebalanced.pct_change()
    print('PCT')
    print(pct.head())
    print('SIGNAL')
    print(signal.head())
    returns = (signal * pct.dropna()).sum(axis=1)
    returns.columns=['Returns']
    return returns

def CumulativeReturns(priceData:pd.DataFrame,signal:pd.DataFrame, rebalancePeriod):
    returns = Returns(priceData,signal,rebalancePeriod)
    cumulativeReturns = (1+returns).cumprod()-1
    cumulativeReturns.columns = ['CumulativeReturns']
    return cumulativeReturns

def Sharpe(priceData:pd.DataFrame,signal:pd.DataFrame,rebalancePeriod):
    returns = Returns(priceData,signal,rebalancePeriod)
    Sharpe = np.round(returns.mean(axis=0) / returns.std(axis=0), 2)
    return Sharpe