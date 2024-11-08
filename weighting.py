import pandas as pd
import numpy as np


class signalWeighter:
    def __init__(self, unweighted:pd.DataFrame,z:pd.DataFrame,momenta:pd.DataFrame):
        self.unweighted = unweighted
        self.z = z
        self.momenta = momenta

    def linear(self,beta):
        """Weight signals linearly in proportion to their z-score"""
        print('BETA = ', beta)
        momenta_sign = self.momenta.apply(np.sign)
        weighted = beta*self.unweighted*self.z.abs()*momenta_sign
        # normalise
        weighted.div(weighted.abs().sum(axis=1), axis=0).apply(np.round)
        return weighted
    def softmax(self,beta:float=1.0):
        print('BETA = ', beta)
        momenta_sign = self.momenta.apply(np.sign)
        softmax = self.z.abs().apply(lambda x: np.exp(beta*x)/np.exp(beta*x).sum(), axis=1)
        weighted = self.unweighted*softmax*momenta_sign

        return weighted

    def getWeightedSignal(self, function, params):
        if function == 'linear':
            beta = params.get('beta',1)
            return self.linear(beta)
        elif function == 'softmax':
            beta = params.get('beta',1)
            return self.softmax(beta)