import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def VIX(z:pd.DataFrame):
    for i in range(1,6):

        z_selected = z[np.random.choice(z.columns)]

        vix_close = pd.read_csv('VIX.csv', index_col='Date')
        vix_close.index = pd.to_datetime(vix_close.index)

        aligned_data = pd.concat([z_selected, vix_close], axis=1).dropna()

        aligned_data.columns = ['zscore', 'VIX']

        print(aligned_data.head())
        aligned_data['Lagged_VIX'] = aligned_data['VIX'].shift(i)
        aligned_data = aligned_data.dropna()
        plt.figure(figsize=(10, 6))
        plt.scatter(aligned_data['Lagged_VIX'], aligned_data['zscore'], alpha=0.5)
        plt.title('Z-scores vs Lagged VIX {}'.format(i))
        plt.xlabel('Lagged VIX')
        plt.ylabel('Z-score')
        plt.grid(True)
        #plt.show()


    return vix_close

def getAllignedExogLags(z:pd.DataFrame,L:int=1)->pd.DataFrame:
    Lags = list(range(1,L+1))
    vix_close = pd.read_csv('VIX.csv', index_col='Date')
    vix_close.index = pd.to_datetime(vix_close.index)
    vix_close = vix_close.reindex(z.index).ffill()
    VIXLags = pd.DataFrame(columns = Lags)


    for l in Lags:
        vix_lag = vix_close.loc[z.index].shift(l)
        VIXLags[l] = vix_lag

    VIXLags = VIXLags.fillna(0)

    VIXExog = VIXLags[[l for l in Lags]].values

    return VIXExog

def getExogLagsupdate(z,L):
    Lags = list(range(1, L + 1))
    vix_close = pd.read_csv('VIX.csv', index_col='Date')
    vix_close.index = pd.to_datetime(vix_close.index)
    VIXLags = pd.DataFrame(columns=Lags)

    for l in Lags:
        vix_lag = vix_close.shift(l)
        VIXLags[l] = vix_lag

    VIXLags = VIXLags.fillna(0)

    return VIXLags[[l for l in Lags]]

