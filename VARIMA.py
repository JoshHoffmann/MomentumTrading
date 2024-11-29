import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX

def VARIMAd(z:pd.DataFrame,alpha:float=0.01,d_space=range(0,5)):
    d_params = {c:None for c in z.columns}
    for c in z.columns:
        test_series = z[c]
        for d in d_space:
            p_value = adfuller(test_series)[1]
            if p_value<alpha:
                d_params[c] = d
                break
            else:
                test_series = test_series.diff().dropna()
        if d_params[c] is None:
            print('Warning: No differencing order was determined for ', c)

    return d_params
def diffData(z:pd.DataFrame,d_params:dict):
    for c in z.columns:
        d = d_params[c]
        z[c] = z[c].diff(d)
    return z

def VARIMApq(z:pd.DataFrame,d_params:dict,p_max:int=5,q_max:int=5):
    model_order = {}
    z = diffData(z,d_params).dropna()

    min_bic = np.inf
    order = None
    for p in range(1, p_max + 1):
        for q in range(1, q_max + 1):
            try:
                model = VARMAX(endog=z, order=(p, q))
                bic = model.fit(disp=False).bic
                print(f"bic: {bic} for p={p}, q={q}")

                if bic < min_bic:
                    min_bic = bic
                    order = (p, q)
            except Exception as e:
                print(f"Error with p={p}, q={q}: {e}")
                continue

    model_order = {'p': order[0], 'q': order[1]}

    return model_order







z = pd.DataFrame({
    'series_1': np.cumsum(np.random.randn(100)),
    'series_2': np.cumsum(np.random.randn(100) * 2),
    'series_3': np.cumsum(np.random.randn(100) * 0.5)
})

d_params = VARIMAd(z)
print(d_params)
models = VARIMApq(z,d_params)
print(models)