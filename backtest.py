import ARIMA
import pandas as pd
import pmdarima
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA as arima
import weighting



class zpThresh:
    '''Strategy that trades p-period momentum above a threshold.'''
    def __init__(self,z,train_window,period=1, threshold=1):
        self.z = z
        self.train_window = train_window
        self.period = period
        self.threshold = threshold

    def simulateSignal(self):
        zp = self.z.loc[self.period,'z']  # Get desired momentum period
        N_end = len(zp)

        models = ARIMA.trainARIMA(zp,self.train_window) # time-series keyed dictionary of ARIMA models
        # Initialise forecasts of z_p momentum
        z_forecast = pd.DataFrame(index=zp.index[self.train_window:], columns=zp.columns)
        z_conf = pd.DataFrame(index=zp.index[self.train_window:], columns=zp.columns)
        # Simulate trade by iterating through days from end of training onwards
        for i in range(self.train_window+1, N_end):
            print('i = {}, N = {}'.format(i,N_end))
            t = zp.index[i] # Get current time step
            # Iterate through stock series and get fitted model
            for c in zp.columns:
                model = models[c]
                model.update(zp.loc[t,c]) # Update model with current observation
                forecast, conf =  model.predict(n_periods=1, return_conf_int=True) # Forecast 1 step ahead
                z_forecast.loc[t,c]= forecast # Save forecast
                z_conf.loc[t,c] = conf
         # This strategy activates trading signals when the z-score is forecasted to be above a threshold
        unweighted = (z_forecast.abs().shift(-1)>self.threshold).astype(int)
        weighted = weighting.signalWeighter(unweighted=unweighted, z=zp).getWeightedSignal('linear')
        return weighted

class Backtest:
    def __init__(self,strategy):
        self.strategy = strategy

    def run(self):
        return self.strategy.simulateSignal()