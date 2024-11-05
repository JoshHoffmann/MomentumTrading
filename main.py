import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fontTools.misc.cython import returns

import backtest
import momenta
import ARIMA
from statsmodels.tsa.arima.model import ARIMA as arima


np.random.seed(1)

# Read daily closing price data of S&P500 stocks into data frame, indexed by date
closeData = pd.read_csv('2020-2024_Data.csv', index_col='date')
closeData.index = pd.to_datetime([date.split()[0] for date in closeData.index]) # Convert index to datetime


closeData.index = pd.DatetimeIndex(closeData.index).to_period('D')

closeData = closeData.iloc[0:500,100:111]

periods = [1,3,6] # Define Momenta periods (months)

Momenta = momenta.getMomentum(closeData, periods) # Get momenta for all periods
Momenta.loc[1,'momentum'].plot()
zscores = momenta.getZScores(Momenta) # Get cross-sectional z-scores for all momenta periods
z = zscores.loc[1,'z']
z.plot()

p_space = list(range(1,5))
q_space = list(range(1,5))


print('CLASS TEST')
s = backtest.zpThresh(z,300,0.05,p_space, q_space)
bt = backtest.Backtest(s)
unweighted = bt.run().dropna()
print(unweighted.head())
plt.figure()
unweighted.plot()
plt.show()
weighted = unweighted*z.loc[unweighted.index,:]
normalised = weighted.div(weighted.abs().sum(axis=1), axis=0).where(weighted.abs().sum(axis=1)!=0, weighted)

prices = closeData.loc[normalised.index,normalised.columns]
r = (normalised.shift(1)*prices.pct_change()).sum(axis=1)

cumulative = (1+r).cumprod()-1

plt.figure()
r.plot()
cumulative.plot()
plt.show()

Sharpe = np.round(r.mean(axis=0)/r.std(axis=0),2)
print('Sharpe Ratio ', Sharpe)




