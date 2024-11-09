import matplotlib.pyplot as plt
import pandas as pd
import backtest
import metrics
import momenta
import plotting

'''pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)'''

# Read daily closing price data of S&P500 stocks into data frame, indexed by date
closeData = pd.read_csv('2020-2024_Data.csv', index_col='date')
closeData.index = pd.to_datetime(closeData.index)

closeData = closeData.iloc[0:400,0:11] # Choose sub set for testing

periods = [1,3,6] # Define Momenta periods (months)

Momenta = momenta.getMomentum(closeData, periods) # Get momenta for all periods
Momenta.loc[1,'momentum'].plot() # Plot 1-month momenta
zscores = momenta.getZScores(Momenta) # Get cross-sectional z-scores for all momenta periods
# Plot all z-scores
for p in periods:
    zscores.loc[p,'z'].plot()
    plt.show()

# Get simulated trading strategy for backtest of zpThresh strategy.
signal = (backtest.Longshort(zscores,Momenta,200,alpha=0.01, rebalancePeriod='W',pre_smoothing='MA',
                              pre_smooth_params={'window':3},weighting_func='linear',filter_func='TopMag',filter_params={'top':5}).
           strategy(strategy='zpThresh',period=1,threshold=1.5))

signal.plot() # Plot trading signal
plt.show()
print('SIGNAL')
print(signal)

# Get metrics
returns = metrics.Returns(closeData,signal,'W')
print('RETURNS')
print(returns)
plotting.plotReturns(returns)
plotting.plotReturnsHist(returns)
cumulative = metrics.CumulativeReturns(closeData,signal,'W')
print('CUMULATIVE')
print(cumulative)
Sharpe = metrics.Sharpe(closeData,signal,'W')
print('Sharpe Ratio ', Sharpe)











