import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import momenta

# Read daily closing price data of S&P500 stocks into data frame, indexed by date
closeData = pd.read_csv('2020-2024_Data.csv', index_col='date')
closeData.index = pd.to_datetime(closeData.index) # Convert index to datetime

periods = [1,3,6]

Momenta = momenta.getMomentum(closeData, periods)
Momenta.loc[1,'momentum'].plot()

zscores = momenta.getZScores(Momenta)
zscores.loc[1,'z'].plot()
plt.show()
