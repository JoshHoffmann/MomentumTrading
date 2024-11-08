""" Handles Backtesting of strategies."""
import ARIMA
import pandas as pd
import weighting
import plotting
import filters


def Rebalance(w:pd.DataFrame, rebalancePeriod:str)->pd.DataFrame:
    '''This function handles rebalancing the trading signal at the desired frequency. It takes a dataframe weighted
    signal w and rebalancing period string. It outputs a dataframe of the rebalanced signals, forward filled between
     rebalancing dates.'''

    # Find rebalancing points based on desried frequency
    rebalancePoints = w.resample(rebalancePeriod).last().index

    # Rebalancing points may not coinccide with trading days. They must be adjusted accordingly
    # Where the rebalancing day is not a trading day, rebalance at closest day before.
    adj_rebalancePoints = [w.index[w.index <= point][-1] for point in rebalancePoints] # Adjusted rebalance points

    w_rebalanced = w.loc[adj_rebalancePoints].reindex(w.index).ffill().dropna() # Forward fill between rebalance points

    return w_rebalanced


class Longshort:
    """ Class of Long-Short strategies.
    z - Master dataframe of z-scores
    momenta - Master dataframe of momenta
    train window - index identifying end of traning data for initial training of the ARIMA model
    rebalancePeriod - string selecting desried portfolio rebalancing frequency i.e. daily ('D'), weekly ('W') etc
    alpha - float selecting critical value for adf test when determining order of integration d for ARIMA models.
    pre_smoothing - string selecting desired data pre-smoothing method (if any). Pre smoothing is applied to data before
    training the ARIMA model. E.g. 'MA' for smoothng by moving average, 'EWW' for smoothing by exponential moving average
    pre_smooth params - dictionary of parameters for pre-smoothing function of the form {'param_name':param}.
     E.g. When pre-smoothing by 'MA' pre_smoothing_params= {'window':5} will smooth by 5 day moving average.
    """
    def __init__(self,z:pd.DataFrame,momenta:pd.DataFrame,train_window:int,rebalancePeriod:str='W',alpha:float=0.05,
                 pre_smoothing=None,pre_smooth_params=None,weighting_func:str='linear',weighting_params=None):
        self.z = z
        self.momenta = momenta
        self.train_window = train_window
        self.rebalancePeriod = rebalancePeriod
        self.alpha = alpha
        self.pre_smoothing = pre_smoothing
        self.pre_smooth_params = pre_smooth_params if pre_smooth_params else {}
        self.weighting_func = weighting_func
        self.weighting_params = weighting_params if weighting_params else {}

    def preSmooth(self,z_pre:pd.DataFrame)->pd.DataFrame:
        """ Handles pre-smoothing of data."""
        if self.pre_smoothing == 'MA':
            window = self.pre_smooth_params.get('window',5) # window defaults to 5
            return z_pre.rolling(window).mean().dropna()
        elif self.pre_smoothing == 'EWM':
            span = self.pre_smooth_params.get('span',3) # span defaults to 3
            return z_pre.ewm(span).mean().dropna()
        else:
            return z_pre.dropna() # Return raw data if no pre-smoothing applied



    def zpThresh(self, period:int=1,threshold:float=1.0)->pd.DataFrame:
        '''zpThresh strategy trades based on the selected p-period momentum being above a threshold. It takes the
         selected momentum period (this must be one of the periods already calculated) and the desired threshold. '''

        zp = self.preSmooth(self.z.loc[period,'z']) # Get desired momentum period
        zp.name = 'z{}'.format(period)

        # Train ARIMA models on initial data
        models = ARIMA.trainARIMA(zp,self.train_window,d_alpha=self.alpha)
        # Initialise forecasts of z_p momentum
        # Important note here: z_forecast[t] will contain the forecast for the NEXT day i.e. what the z-score will be
        # at t+1 day ahead. The reason for this is to make activating the trading signal a bit easier which can be seen
        # below
        z_forecast = pd.DataFrame(index=zp.index[self.train_window:], columns=zp.columns)
        z_forecast.name = 'z{} forecast'.format(period)
        z_conf = pd.DataFrame(index=zp.index[self.train_window:], columns=zp.columns)

        # Simulate trade by iterating through days from end of training onwards
        T = zp.iloc[self.train_window:,:].index # Get out of sample time steps to iterate through
        for t in T:
            print('t = {}, T = {}'.format(t,T[-1]))
            # Iterate through stock series and get fitted models
            for c in zp.columns:
                model = models[c] # Retrieve fitted model
                model.update(zp.loc[t,c]) # Update model with current observation
                m = model.predict(n_periods=1, return_conf_int=True) # Forecast 1 step ahead
                forecast, conf =  m[0][0], m[1][0]  # Get forcast and conf intervals (not used currently)
                z_forecast.loc[t,c]= forecast # Save forecast
                z_conf.loc[t,c] = conf

        plotting.plotForecast(zp.shift(-1),z_forecast) # Call plotting function to ranndomly select a stock and plot
        # its observed and forecasted z score. Observed z scores need to be shifted to align with forecast dataframe

        # This strategy activates trading signals when the z-score is forecasted to be above a threshold
        unweighted = (z_forecast.abs()>threshold).astype(int) # Activate signals acording to trading logic
        test = filters.filter(zp,unweighted).filterFunction('TopMag',top=15)
        unweighted=test
        # Weight signals according to selected weighting
        weighted = (weighting.signalWeighter(unweighted=unweighted, z=zp,momenta=self.momenta.loc[period,'momentum']).
                    getWeightedSignal(self.weighting_func, self.weighting_params))
        print('Rebalance by ', self.rebalancePeriod)
        print('weighted')
        print(weighted.head(50))
        rebalancedSignal = Rebalance(weighted,self.rebalancePeriod) # Rebalance signal according to desired frequency
        print('Rebalanced')
        print(rebalancedSignal.head(50))

        return rebalancedSignal

    def CrossOver(self,fast:int,slow:int)->pd.DataFrame:
        """ CrossOver strategy trades based on the selected fast momentum is forecasted to cross the selected slow
        momentum. It takes the selected momentum periods (these must be periods already calculated)"""
        # Retrieve fast and slow momenta
        zslow = self.preSmooth(self.z.loc[slow, 'z'])
        zslow.name = 'z-slow'
        zfast = self.preSmooth(self.z.loc[fast, 'z']).reindex(zslow.index)
        zfast.name = 'z-fast'

        # Train initial ARIMA models for fast and slow momenta
        fastmodels = ARIMA.trainARIMA(zfast, self.train_window,
                                  d_alpha=self.alpha)  # time-series keyed dictionary of ARIMA models
        slowmodels = ARIMA.trainARIMA(zslow, self.train_window,)

        # Initialise forecast dataframes
        zfast_forecast = pd.DataFrame(index=zfast.index[self.train_window:], columns=zfast.columns)
        zfast_forecast.name='zfast-forecast'
        zslow_forecast = pd.DataFrame(index=zslow.index[self.train_window:], columns=zslow.columns)
        zslow_forecast.name='zslow-forecast'

        T = zslow.iloc[self.train_window:, :].index  # Get out of sample time steps to iterate through
        # Simulate trade by iterating through days from end of training onwards
        for t in T:
            print('t = {}, T = {}'.format(t,T[-1]))
            # Iterate through stock series and get fitted models
            for c in zfast.columns:
                fastmodel = fastmodels[c]
                slowmodel = slowmodels[c]

                # Update models with current observation
                fastmodel.update(zfast.loc[t, c])
                slowmodel.update(zslow.loc[t, c])

                # Get step ahead forecasts
                mfast = fastmodel.predict(n_periods=1, return_conf_int=True)
                mslow = slowmodel.predict(n_periods=1, return_conf_int=True)
                fastforecast, conf = mfast[0][0], mfast[1][0]
                slowforecast, slowconf = mslow[0][0], mslow[1][0]

                # Store forecasts
                zfast_forecast.loc[t, c] = fastforecast
                zslow_forecast.loc[t, c] = slowforecast

        plotting.plotForecast(zfast.shift(-1), zfast_forecast)
        plotting.plotForecast(zslow.shift(-1), zslow_forecast)
        # This strategy activates trading signals when the zfast>zslow
        unweighted = (zfast_forecast.abs() > zslow_forecast.abs()).astype(int)
        weighted = weighting.signalWeighter(unweighted=unweighted, z=zfast,momenta=self.momenta.loc[fast,'momentum']).getWeightedSignal(self.weighting_func,self.weighting_params)
        rebalancedSignal = Rebalance(weighted, self.rebalancePeriod)
        return rebalancedSignal

    def strategy(self,strategy='zpThresh',**kwargs):
        if strategy == 'zpThresh':
            return self.zpThresh(**kwargs)
        elif strategy == 'CrossOver':
            return self.CrossOver(**kwargs)
