import pandas as pd
import numpy as np
import plotting
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)

def priceRebalance(priceData,rebalancePeriod):
    rebalancePoints = priceData.resample(rebalancePeriod).last().index

    adj_rebalancePoints = [priceData.index[priceData.index <= point][-1] for point in rebalancePoints]

    w_rebalanced = priceData.loc[adj_rebalancePoints].reindex(priceData.index).ffill().dropna()

    return w_rebalanced, adj_rebalancePoints

def Returns(priceData:pd.DataFrame,signal:pd.DataFrame,rebalancePeriod:str,t=True):

    prices_rebalanced, rebalancePoints = priceRebalance(priceData.loc[signal.index],rebalancePeriod)

    pct = prices_rebalanced.loc[rebalancePoints].pct_change().fillna(0)

    returns = (pct * signal.loc[rebalancePoints]).sum(axis=1)

    return returns

def CumulativeReturns(priceData:pd.DataFrame,signal:pd.DataFrame, rebalancePeriod):
    returns = Returns(priceData,signal,rebalancePeriod,t=False)
    cumulativeReturns = (1+returns).cumprod()-1
    cumulativeReturns.columns = ['CumulativeReturns']
    plotting.plotCumulative(cumulativeReturns)
    return cumulativeReturns

def Sharpe(priceData:pd.DataFrame,signal:pd.DataFrame,rebalancePeriod):
    returns = Returns(priceData,signal,rebalancePeriod,t=False)
    Sharpe = np.round(returns.mean(axis=0) / returns.std(axis=0), 2)
    return Sharpe

