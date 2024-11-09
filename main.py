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

closeData = closeData.iloc[0:350,0:9] # Choose sub set for testing

periods = [1,3,6] # Define Momenta periods (months)

Momenta = momenta.getMomentum(closeData, periods) # Get momenta for all periods
Momenta.loc[1,'momentum'].plot() # Plot 1-month momenta
zscores = momenta.getZScores(Momenta) # Get cross-sectional z-scores for all momenta periods
# Plot all z-scores
for p in periods:
    zscores.loc[p,'z'].plot()
    plt.show()


signal = (backtest.Longshort(zscores,closeData,Momenta,200,alpha=0.01, rebalancePeriod='W',pre_smoothing='EWM',
                              pre_smooth_params={'span':10},weighting_func='softmax',weighting_params={'beta':1},
                                filter_func='vol', filter_params={'priceData':closeData,'window':40}).
           strategy(strategy='CrossOver',fast=1,slow=3))

signal.plot() # Plot trading signal
plt.show()















