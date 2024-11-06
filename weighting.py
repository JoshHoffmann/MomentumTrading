import pandas as pd
import numpy as np


class signalWeighter:
    def __init__(self, unweighted:pd.DataFrame,z:pd.DataFrame):
        self.unweighted = unweighted
        self.z = z

    def linear(self,beta=1):
        """Weight signals linearly in proportion to their z-score"""
        weighted = beta*self.unweighted*self.z
        # normalise
        weighted = np.round(weighted.div(weighted.abs().sum(axis=1), axis=0),2)
        return weighted

    def getWeightedSignal(self, function, **kwargs):
        if function == 'linear':
            return self.linear(**kwargs)
