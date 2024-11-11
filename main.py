import matplotlib.pyplot as plt
import pandas as pd
import backtest
import momenta
import plotting


# Read daily closing price data of S&P500 stocks into data frame, indexed by date
closeData = pd.read_csv('2020-2024_Data.csv', index_col='date')
closeData.index = pd.to_datetime(closeData.index)

closeData = closeData.iloc[0:320,0:29] # Choose sub set for testing

periods = [1,3,6] # Define Momenta periods (months)

Momenta = momenta.getMomentum(closeData, periods) # Get momenta for all periods
plotting.plotMomenta(Momenta,periods) # Plot Momenta
zscores = momenta.getZScores(Momenta) # Get cross-sectional z-scores for all momenta periods

plotting.plotZScores(zscores,periods)

signal1 = (backtest.Longshort(zscores,closeData,Momenta,200,alpha=0.01, rebalancePeriod='W',pre_smoothing='MA',
                              pre_smooth_params={'window':4},weighting_func='volAdjust',weighting_params={'window':30},
                                filter_func='TopMag', filter_params={'top':10}).
           strategy(strategy='zpThresh',period=1,threshold=5,sweep=True,period_space=[1,3,6],thresh_space=[1,2,3,4]))

signal1.plot() # Plot trading signal
plt.show()