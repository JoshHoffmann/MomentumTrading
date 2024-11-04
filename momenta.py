import pandas as pd

# Define a function to calculate p-period momenta
def getPMomentum(priceData:pd.DataFrame, p:int):
    """This function calculates p-period momentum from stock price data. It takes a pandas dataFrame of price time
    series and an integer p (the period in months). It returns a dataFrame of the momenta time series."""
    return priceData - priceData.shift(22*p)

def getPZScores(ts:pd.DataFrame):
    """This function calculates the time series of cross-sectional z-scores of time series data.
    It takes as input a dataFrame containing 1 or more time series. It returns a dataFrame of the cross-sectional
     z-scores."""
    mean = ts.mean(axis=1)
    std = ts.std(axis=1)
    return ts.sub(mean, axis = 0).div(std,axis=0)
def getMomentum(priceData:pd.DataFrame, periods:list):
    """This function retrieves momenta time series for a multiple momentum time series. It takes a pandas dataFrame of
    price time series and a list of momenta periods. It returns a period-indexed dataFrame of the momenta time series"""

    # Define a period-keyed dictionary containing p-period momenta dataFrames
    momentum_dict = {p: getPMomentum(priceData,p) for p in periods}
    # Store all p-period momenta df's in a master data frame, indexed by period
    momentum_df = pd.DataFrame(list(momentum_dict.items()), columns=['periods', 'momentum']).set_index('periods')
    return momentum_df

def getZScores(Momentum_df:pd.DataFrame):
    """This function calculates cross-sectional z-scores for a collection of p-period momentum time series. It takes as
    input a period-indexed master data frame containing p-period momenta time series. It outputs a period-index master
     data frame of p-period momenta cross-sectional z-score time series"""
    periods = list(Momentum_df.index) # Retrieve periods from dataFrame index
    # Define a period-keyed dictionary to store p-period momentum cross-sectional z-score time series
    z_dict = {p:getPZScores(Momentum_df.loc[p,'momentum']) for p in periods}
    # Convert dictionary to period-indexed master dataFrame
    z_df = pd.DataFrame(list(z_dict.items()), columns=['periods', 'z']).set_index('periods')
    return z_df