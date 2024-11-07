import ARIMA
import pandas as pd
import weighting
import plotting


def Rebalance(w, rebalancePeriod):
    rebalancePoints = w.resample(rebalancePeriod).last().index

    adj_rebalancePoints = [w.index[w.index <= point][-1] for point in rebalancePoints]

    w_rebalanced = w.loc[adj_rebalancePoints].reindex(w.index).ffill().dropna()

    return w_rebalanced


class Longshort:
    '''Strategy that trades p-period momentum above a threshold.'''
    def __init__(self,z,momenta,train_window,rebalancePeriod='W',alpha=0.05,pre_smoothing=None,pre_smooth_params=None):
        self.z = z
        self.momenta = momenta
        self.train_window = train_window
        self.rebalancePeriod = rebalancePeriod
        self.alpha = alpha
        self.pre_smoothing = pre_smoothing
        self.pre_smooth_params = pre_smooth_params if pre_smooth_params else {}

    def preSmooth(self,z_pre):
        if self.pre_smoothing == 'MA':
            window = self.pre_smooth_params.get('window',5)
            return z_pre.rolling(window).mean().dropna()
        elif self.pre_smoothing == 'EWM':
            span = self.pre_smooth_params.get('span',3)
            return z_pre.ewm(span).mean().dropna()
        else:
            return z_pre.dropna()

    def zpThresh(self, period=1,threshold=1):
        zp = self.preSmooth(self.z.loc[period,'z']) # Get desired momentum period
        N_end = len(zp)

        models = ARIMA.trainARIMA(zp,self.train_window,d_alpha=self.alpha) # time-series keyed dictionary of ARIMA models
        # Initialise forecasts of z_p momentum
        z_forecast = pd.DataFrame(index=zp.index[self.train_window:], columns=zp.columns)
        z_conf = pd.DataFrame(index=zp.index[self.train_window:], columns=zp.columns)
        # Simulate trade by iterating through days from end of training onwards
        zp_obs = zp.iloc[self.train_window:,:]
        T = zp_obs.index
        for t in T:
            print('t = {}, T = {}'.format(t,T[-1]))
            # Iterate through stock series and get fitted model
            for c in zp.columns:
                model = models[c]
                model.update(zp.loc[t,c]) # Update model with current observation
                m = model.predict(n_periods=1, return_conf_int=True)
                forecast, conf =  m[0][0], m[1][0]  # Forecast 1 step ahead
                z_forecast.loc[t,c]= forecast # Save forecast
                z_conf.loc[t,c] = conf
         # This strategy activates trading signals when the z-score is forecasted to be above a threshold
        plotting.plotForecast(zp,z_forecast)
        unweighted = (z_forecast.abs()>threshold).astype(int)
        weighted = weighting.signalWeighter(unweighted=unweighted, z=zp,momenta=self.momenta.loc[period,'momentum']).getWeightedSignal('linear')
        rebalancedSignal = Rebalance(weighted,self.rebalancePeriod)

        return rebalancedSignal

    def crossOver(self,fast,slow):
        zslow = self.preSmooth(self.z.loc[slow, 'z'])
        zfast = self.preSmooth(self.z.loc[fast, 'z']).reindex(zslow.index)

        fastmodels = ARIMA.trainARIMA(zfast, self.train_window,
                                  d_alpha=self.alpha)  # time-series keyed dictionary of ARIMA models
        slowmodels = ARIMA.trainARIMA(zslow, self.train_window,)

        zfast_forecast = pd.DataFrame(index=zfast.index[self.train_window:], columns=zfast.columns)
        zslow_forecast = pd.DataFrame(index=zslow.index[self.train_window:], columns=zslow.columns)
        # Simulate trade by iterating through days from end of training onwards
        N_end = len(zslow)

        for i in range(self.train_window, N_end - 1):
            print('i = {}, N = {}'.format(i, N_end))
            t = zfast.index[i]  # Get current time step
            # Iterate through stock series and get fitted model
            for c in zfast.columns:
                fastmodel = fastmodels[c]
                slowmodel = slowmodels[c]

                fastmodel.update(zfast.loc[t, c])  # Update model with current observation
                slowmodel.update(zslow.loc[t, c])

                mfast = fastmodel.predict(n_periods=1, return_conf_int=True)
                mslow = slowmodel.predict(n_periods=1, return_conf_int=True)
                fastforecast, conf = mfast[0][0], mfast[1][0]  # Forecast 1 step ahead
                slowforecast, slowconf = mslow[0][0], mslow[1][0]

                zfast_forecast.loc[t, c] = fastforecast  # Save forecast
                zslow_forecast.loc[t, c] = slowforecast

        # This strategy activates trading signals when the zfast>zslow
        unweighted = (zfast_forecast.abs() > zslow_forecast.abs()).astype(int)
        weighted = weighting.signalWeighter(unweighted=unweighted, z=zfast,momenta=self.momenta.loc[fast,'momentum']).getWeightedSignal('linear')
        rebalancedSignal = Rebalance(weighted, self.rebalancePeriod)
        return rebalancedSignal

    def strategy(self,strategy='zpThresh',**kwargs):
        if strategy == 'zpThresh':
            return self.zpThresh(**kwargs)
        elif strategy == 'crossOver':
            return self.crossOver(**kwargs)
