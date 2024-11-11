import pandas as pd
import numpy as np
import plotting


def RebalanceReturns(priceData:pd.DataFrame, rebalancePeriod:str):

    # Find rebalancing points based on desried frequency
    rebalancePoints = priceData.resample(rebalancePeriod).last().index

    # Rebalancing points may not coinccide with trading days. They must be adjusted accordingly
    # Where the rebalancing day is not a trading day, rebalance at closest day before.
    adj_rebalancePoints = [priceData.index[priceData.index <= point][-1] for point in rebalancePoints]  # Adjusted rebalance points

    return adj_rebalancePoints


def Returns(priceData:pd.DataFrame,signal:pd.DataFrame,rebalancePeriod:str):
    rebalancePoints = RebalanceReturns(signal,rebalancePeriod)
    price_rebalance = priceData.loc[rebalancePoints]
    pct = price_rebalance.pct_change().fillna(0)
    print('PCT')
    print(pct.head())
    returns = (pct* signal.loc[rebalancePoints]).sum(axis=1)

    return returns

def CumulativeReturns(priceData:pd.DataFrame,signal:pd.DataFrame, rebalancePeriod)->pd.DataFrame:
    returns = Returns(priceData,signal,rebalancePeriod)
    cumulativeReturns = (1+returns).cumprod()-1
    cumulativeReturns.columns = ['CumulativeReturns']
    plotting.plotCumulative(cumulativeReturns)
    return cumulativeReturns

def Sharpe(priceData:pd.DataFrame,signal:pd.DataFrame,rebalancePeriod)->float:
    returns = Returns(priceData,signal,rebalancePeriod)
    Sharpe = np.round(returns.mean(axis=0) / returns.std(axis=0), 2)
    return Sharpe

def maxdrawdown(cumulativeReturns:pd.DataFrame):
    max_point = cumulativeReturns.cummax()  # Peak of cumulative returns for calculating drawdown

    drawdown = (cumulativeReturns - max_point) / max_point
    # If peak is zero, we can get a division by zero error in calculation. Set drawdown equal to zero where this happens
    drawdown[drawdown == -np.inf] = 0
    max_draw_down = np.round(drawdown.min(), 2)

    return max_draw_down