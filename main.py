import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import backtest
import metrics
import momenta
import ARIMA
from statsmodels.tsa.arima.model import ARIMA as arima


np.random.seed(1)

# Read daily closing price data of S&P500 stocks into data frame, indexed by date
closeData = pd.read_csv('2020-2024_Data.csv', index_col='date')
closeData.index = pd.to_datetime([date.split()[0] for date in closeData.index]) # Convert index to datetime


closeData.index = pd.DatetimeIndex(closeData.index).to_period('D')

closeData = closeData.iloc[0:400,100:103]

periods = [1,3,6] # Define Momenta periods (months)

Momenta = momenta.getMomentum(closeData, periods) # Get momenta for all periods
Momenta.loc[1,'momentum'].plot()
zscores = momenta.getZScores(Momenta) # Get cross-sectional z-scores for all momenta periods
for p in periods:
    zscores.loc[p,'z'].plot()
    plt.show()


s = backtest.zpThresh(zscores,train_window=300,period=1)
bt = backtest.Backtest(s)
signal = bt.run().dropna()
print(signal.head())

signal.plot()
plt.show()


returns = metrics.Returns(closeData,signal)
cumulative = metrics.CumulativeReturns(closeData,signal)
Sharpe = metrics.Sharpe(closeData,signal)
print('Sharpe Ratio ', Sharpe)



plt.figure()
returns.plot()
cumulative.plot()
plt.show()





