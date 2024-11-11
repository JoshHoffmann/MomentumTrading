import pandas as pd
import numpy as np


class signalWeighter:
    def __init__(self, unweighted:pd.DataFrame,z:pd.DataFrame,momenta:pd.DataFrame,priceData:pd.DataFrame):
        self.unweighted = unweighted
        self.z = z
        self.momenta = momenta
        self.priceData = priceData

    def linear(self,beta:float)->pd.DataFrame:
        """Weight signals linearly in proportion to their z-score"""
        print('BETA = ', beta)
        I = self.unweighted.index
        momenta_sign = self.momenta.loc[I,:].apply(np.sign)

        weighted = beta*self.unweighted*self.z.abs()*momenta_sign
        # normalise
        weighted = weighted.div(weighted.abs().sum(axis=1), axis=0)
        return weighted
    def softmax(self,beta:float=1.0):
        print('BETA = ', beta)
        momenta_sign = self.momenta.apply(np.sign)
        softmax = self.z.abs().apply(lambda x: np.exp(beta*x)/np.exp(beta*x).sum(), axis=1)
        weighted = self.unweighted*softmax*momenta_sign

        return weighted

    def volAdjust(self,window:int)->pd.DataFrame:
        I = self.unweighted.index
        momenta_sign = self.momenta.loc[I, :].apply(np.sign)
        vol = self.priceData.pct_change().loc[I,:].rolling(window=window).std()
        weighted = self.unweighted*self.z.abs()*momenta_sign/vol
        weighted = weighted.div(weighted.abs().sum(axis=1), axis=0)
        return weighted

    def volProp(self,window:int)->pd.DataFrame:
        I = self.unweighted.index
        momenta_sign = self.momenta.loc[I, :].apply(np.sign)
        vol = self.priceData.pct_change().loc[I, :].rolling(window=window).std()
        weighted = self.unweighted * self.z.abs() * momenta_sign * vol
        weighted = weighted.div(weighted.abs().sum(axis=1), axis=0)
        return weighted

    def getWeightedSignal(self, function:str, params:dict)->pd.DataFrame:
        if function == 'linear':
            beta = params.get('beta',1)
            return self.linear(beta)
        elif function == 'softmax':
            beta = params.get('beta',1)
            return self.softmax(beta)
        elif function == 'volAdjust':
            priceData = params.get('priceData')
            window = params.get('window',33)
            return self.volAdjust(window)
        elif function=='volProp':
            priceData = params.get('priceData')
            window = params.get('window',33)
            return self.volProp(window)