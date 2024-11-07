import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea

def plotReturns(returns):

    plt.figure()
    returns.plot()
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.title('Strategy Returns')
    plt.show()

    plotReturnsHist(returns)

def plotReturnsHist(returns):
    mean  = returns.mean()
    plt.figure(figsize=(10, 6))

    sea.histplot(returns, bins=30, kde=True, color="skyblue", stat="density")
    plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')

    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.title('Distribution of Strategy Returns')
    plt.legend()
    plt.show()

def plotCumulative(cumulative):
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
    plt.title('{} Observed vs Forecast'.format(obs.name))
    plt.legend()
    plt.show()


