import ARIMA
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA as arima



class zpThresh:
    def __init__(self,z,train_window,d_alpha,p_space,q_space):
        self.z = z
        self.train_window = train_window
        self.d_alpha = d_alpha
        self.p_space = p_space
        self.q_space = q_space

    def simulateSignal(self):
        z_train=self.z.iloc[0:self.train_window]
        orders = ARIMA.getARIMAParams(z_train,self.d_alpha,self.p_space,self.q_space)
        z_forecast = pd.DataFrame(index=self.z.index[self.train_window:], columns=self.z.columns)
        signals = pd.DataFrame(index=self.z.index[self.train_window:], columns=self.z.columns).fillna(0)
        for i in range(self.train_window, len(self.z)):
            t = self.z.index[i]
            trainData = self.z.iloc[0:i]
            # fit the model for each series
            for c in trainData.columns:
                model = arima(trainData[c], order=orders[c])
                fit = model.fit()
                forecast = fit.forecast(steps=1)
                z_forecast.loc[t, c] = forecast[0]
        self.unweighted = (z_forecast.abs().shift(-1)>z_forecast.abs()).astype(int)*self.z
        return self.unweighted.dropna()

class Backtest:
    def __init__(self,strategy):
        self.strategy = strategy

    def run(self):
        return self.strategy.simulateSignal()