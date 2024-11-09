import pandas as pd
import numpy as np
import plotting
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)


def Returns(priceData:pd.DataFrame,signal:pd.DataFrame,rebalancePeriod:str,t=True):

    pct = priceData.pct_change().fillna(0)
    returns = (pct.loc[signal.index] * signal).sum(axis=1)
    print('RETURNS')
    print(returns)

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

