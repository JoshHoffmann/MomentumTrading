import matplotlib.pyplot as plt
import pandas as pd
import backtest
import momenta
import plotting
import exog

#TODO: Update docstrings for all modules
# Maybe move momentum calculations into backtest rather than putting in momenta as an input so to that the strategy
# class already knows what momenta periods have been calculated

#TODO: Begin implementatiom of Monte Carlo backtest

# Read daily closing price data of S&P500 stocks into data frame, indexed by date
closeData = pd.read_csv('2020-2024_Data.csv', index_col='date')
closeData.index = pd.to_datetime(closeData.index)

closeData = closeData.iloc[0:600,0:399] # Choose sub set for testing

periods = [1,3,6] # Define Momenta periods (months)

Momenta = momenta.getMomentum(closeData, periods) # Get momenta for all periods
plotting.plotMomenta(Momenta,periods) # Plot Momenta
zscores = momenta.getZScores(Momenta) # Get cross-sectional z-scores for all momenta periods

plotting.plotZScores(zscores,periods)




'''signal2 = (backtest.Longshort(zscores,closeData,Momenta,200,alpha=0.01, rebalancePeriod='W',pre_smoothing='MA',
                              pre_smooth_params={'window':7},weighting_func='linear',weighting_params={'beta':1}).
           strategy(strategy='zpThresh',period=6,threshold=0.1, sweep=True, period_space=[1,3,6],thresh_space=[1,2,3,4,5,6,7,8,9,10]))'''



