import pandas as pd
import numpy as np
from math import *
from scipy import stats
import matplotlib.pyplot as plt
from random import randint


class table():
    
    def __init__(self, df, datacol=None):
        self.table = df
        if bool(datacol):
            self.table['DATA'] = df[datacol].values
        super(table, self).__init__()

    def add_col(self, colname, groups, groupcol='GROUP'):
        self.table[colname] = [0 for _ in range(len(self.table))]
        for i in groups:
            s = self.table['DATA'].loc[self.table[groupcol].isin(i)]
            vals = np.array(s.values.tolist())
            self.table[colname].loc[s.index] = np.mean(vals)

    def residual(self, col1, col2):
        data = [self.table[col1].loc[i] - self.table[col2].loc[i] for i in self.table.index]
        data2 = [i ** 2 for i in data]
        self.table[col1+'.e'], self.table[col1+'.e^2'] = data, data2
        return np.sum(np.array(data2))

    def bic_table(self, cols):
        data = []
        N = len(self.table)
        for col in cols:
            k = len(self.table[col].unique())
            SSE = np.sum(np.array(self.table[col+'.e^2'].values.tolist()))
            data.append([col, k, SSE, (N*np.log((SSE/N)))+(k*np.log(N))])
        return pd.DataFrame(np.array(data).reshape(-1, 4), columns=['source', 'k', 'SSE', 'BIC'])

