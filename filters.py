import pandas as pd
class filter:
    def __init__(self,z:pd.DataFrame, unweighted):
        self.z = z
        self.unweighted = unweighted

    def filterFunction(self,filter_func='TopMag', **kwargs):
        if filter_func=='TopMag':
            return self.TopMag(**kwargs)
        if filter_func=='vol':
            return self.vol(**kwargs)

    def TopMag(self,top:int=3):
        print('top = ', top)
        for index, row in self.z.iterrows():
            ranked = row.abs().sort_values(ascending=False)  # At each datetime, rank in descending by |z|
            selectedTickers = ranked.head(top).index.tolist()  # Select top stocks, tuned by 'top' parameter
            self.unweighted.loc[index, (~self.unweighted.columns.isin(selectedTickers))] = 0
        return self.unweighted

    def vol(self,price:pd.DataFrame,window:int=22,top:int=5,high:bool=False):
            vol = price.rolling(window=window).std()
            for index, row in vol.iterrows():
                ranked = row.sort_values(ascending=high)
                selectedTickers = ranked.head(top).index.tolist()
                self.unweighted.loc[index, (~self.unweighted.columns.isin(selectedTickers))] = 0
                return self.unweighted