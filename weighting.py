import pandas as pd
import numpy as np


class signalWeighter:
    def __init__(self, unweighted:pd.DataFrame,z:pd.DataFrame,momenta:pd.DataFrame):
        self.unweighted = unweighted
        self.z = z
        self.momenta = momenta

    def linear(self,beta=1):
        """Weight signals linearly in proportion to their z-score"""
        weighted = beta*self.unweighted*self.z.abs()*self.momenta.apply(np.sign)
        print(self.momenta.apply(np.sign).dropna().head())
        # normalise
        weighted.div(weighted.abs().sum(axis=1), axis=0).apply(np.round)
        return weighted

    def getWeightedSignal(self, function, **kwargs):
        if function == 'linear':
            return self.linear(**kwargs)
