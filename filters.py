import pandas as pd
class filter:
    def __init__(self,z:pd.DataFrame, unweighted):
        self.z = z
        self.unweighted = unweighted

    def filterFunction(self,filter_func,params):
        if filter_func=='TopMag':
            return self.TopMag(params)
        if filter_func=='vol':
            return self.vol(params)
        else:
            return self.unweighted

    def TopMag(self,params):
        top = params.get('top',5)
        print('top = ', top)

        for index, row in self.z.iterrows():
            ranked = row.abs().sort_values(ascending=False)  # At each datetime, rank in descending by |z|
            selectedTickers = ranked.head(top).index.tolist()  # Select top stocks, tuned by 'top' parameter
            self.unweighted.loc[index, :] = self.unweighted.columns.isin(selectedTickers).astype(int)
        return self.unweighted

    def vol(self,params):
            priceData = params.get('priceData')
            window = params.get('window',66)
            print('window = ', window)
            top = params.get('top',5)
            high = params.get('high', False)
            vol = priceData.pct_change().fillna(0).rolling(window=window).std()
            for index, row in vol.iterrows():
                ranked = row.sort_values(ascending=high)
                selectedTickers = ranked.head(top).index.tolist()
                self.unweighted.loc[index, (~self.unweighted.columns.isin(selectedTickers))] = 0
            print('Successfully vol weighted')
            return self.unweighted
