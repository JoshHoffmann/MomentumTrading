import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import momenta
import ARIMA
from statsmodels.tsa.arima.model import ARIMA as arima
np.random.seed(1)

# Read daily closing price data of S&P500 stocks into data frame, indexed by date
closeData = pd.read_csv('2020-2024_Data.csv', index_col='date')
closeData.index = pd.to_datetime([date.split()[0] for date in closeData.index]) # Convert index to datetime


closeData.index = pd.DatetimeIndex(closeData.index).to_period('D')

closeData = closeData.iloc[:,100:103]

periods = [1,3,6] # Define Momenta periods (months)

Momenta = momenta.getMomentum(closeData, periods) # Get momenta for all periods
Momenta.loc[1,'momentum'].plot()
zscores = momenta.getZScores(Momenta) # Get cross-sectional z-scores for all momenta periods
zscores.loc[1,'z'].plot()

p_space = list(range(1,5))
q_space = list(range(1,5))
orders = ARIMA.getARIMAParams(zscores.loc[1,'z'],0.05,p_space, q_space)
print(orders)

ARIMA.getPredictions(zscores.loc[1,'z'],orders)


