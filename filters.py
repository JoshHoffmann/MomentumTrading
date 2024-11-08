import pandas as pd
class filter:
    def __init__(self,z:pd.DataFrame, unweighted):
        self.z = z
        self.unweighted = unweighted

    def filterFunction(self,filter_func='TopMag', **kwargs):
        if filter_func=='TopMag':
            return self.TopMag(**kwargs)

    def TopMag(self,top:int=3):
        print('top = ', top)
        for index, row in self.z.iterrows():
            ranked = row.abs().sort_values(ascending=False)  # At each datetime, rank in descending by |z|
            selectedTickers = ranked.head(top).index.tolist()  # Select top stocks, tuned by 'top' parameter
            self.unweighted.loc[index, (~self.unweighted.columns.isin(selectedTickers))] = 0
        return self.unweighted