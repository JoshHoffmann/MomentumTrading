import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea


def plotMomenta(momenta:pd.DataFrame,periods:list[int]):
    for p in periods:
        momenta.loc[p,'momentum'].plot()
        plt.xlabel('Time')
        plt.ylabel('Momentum')
        plt.title('{}-Period Momentum'.format(p))
        plt.show()
def plotZScores(zscores:pd.DataFrame,periods:list[int]):
    for p in periods:
        zscores.loc[p, 'z'].plot()
        plt.xlabel('Time')
        plt.ylabel('z')
        plt.title('{}-Period Z-Score'.format(p))
        plt.show()

def plotReturns(returns:pd.DataFrame):

    plt.figure()
    returns.plot()
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.title('Strategy Returns')
    plt.show()


def plotReturnsHist(returns:pd.DataFrame):
    mean  = returns.mean()
    plt.figure(figsize=(10, 6))

    sea.histplot(returns, bins=60, color="skyblue", stat="density")
    plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')

    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.title('Distribution of Strategy Returns')
    plt.legend()
    plt.show()

def plotCumulative(cumulative:pd.DataFrame):
    plt.figure()
    cumulative.plot()
    plt.xlabel('Time')
    plt.ylabel('Cumulative Returns')
    plt.title('Strategy Cumulative Returns')
    plt.show()

def plotForecast(obs:pd.DataFrame,forecast:pd.DataFrame,n=1):
    selected = np.random.choice(obs.columns, n)
    T = forecast.index
    '''obs.loc[T,selected].plot(label='obs')
    forecast[selected].plot(label='forecast')'''
    plt.plot(obs.loc[T,selected].values, label='{} observed z-score'.format(selected))
    plt.plot(forecast[selected].values, label='{} forecast z-score'.format(selected))
    plt.title('Observed vs Forecast')
    plt.legend()
    plt.show()

def plotzThresh(z:pd.DataFrame,threshold:float):
    zthresh = z.loc[:,(z.dropna().abs()>threshold).any()]
    zthresh.plot()
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'+{threshold}')
    plt.axhline(y=-threshold, color='r', linestyle='--', label=f'-{threshold}')
    plt.xlabel('Time')
    plt.ylabel('z-score')
    plt.legend()
    plt.show()

