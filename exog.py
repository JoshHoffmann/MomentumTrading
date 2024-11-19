import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

def VIX(priceData:pd.DataFrame,z:pd.DataFrame):
    for i in range(1,6):

        z_selected = z[np.random.choice(z.columns)]
        # Download VIX data (you can adjust the start and end dates as needed)
        vix_data = yf.download('^VIX', start='2020-01-01', end='2024-01-01')

        # Display the first few rows of the data
        print(vix_data.head())

        # Extract only the 'Close' column from the VIX data
        vix_close = vix_data['Close']

        # Resample the VIX data to daily frequency, filling missing data if necessary (if your z-scores are daily)
        vix_close = vix_close.resample('D').ffill()  # Forward fill missing values

        # Align with z-scores data
        aligned_data = pd.concat([z_selected, vix_close], axis=1).dropna()

        # Rename columns for clarity
        aligned_data.columns = ['zscore', 'VIX']

        # Display the first few rows of the aligned data
        print(aligned_data.head())
        aligned_data['Lagged_VIX'] = aligned_data['VIX'].shift(i)
        aligned_data = aligned_data.dropna()
        plt.figure(figsize=(10, 6))
        plt.scatter(aligned_data['Lagged_VIX'], aligned_data['zscore'], alpha=0.5)
        plt.title('Z-scores vs Lagged VIX {}'.format(i))
        plt.xlabel('Lagged VIX')
        plt.ylabel('Z-score')
        plt.grid(True)
        plt.show()

    return vix_close