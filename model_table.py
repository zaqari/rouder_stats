import pandas as pd
import numpy as np
from math import *
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from random import randint
from statsmodels.formula.api import ols


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

    def bic_table(self, cols, override_params=None):
        data = []
        N = len(self.table)
        for i, col in enumerate(cols):
            k = len(self.table[col].unique())
            if override_params:
                k=override_params[i]
            SSE = np.sum(np.array(self.table[col+'.e^2'].values.tolist()))
            data.append([col, k, SSE, (N*np.log((SSE/N)))+(k*np.log(N))])
        return pd.DataFrame(np.array(data).reshape(-1, 4), columns=['source', 'k', 'SSE', 'BIC'])

    def regression_slopes(self, cols, y_col, verbose=0):
        x = np.vstack([self.table[col].values for col in cols]).T
        y = self.table[y_col].values

        regression_slopes = np.linalg.lstsq(x,y)

        alpha = y[0] - np.dot(x[0], regression_slopes[0])
        if verbose == 2:
            return regression_slopes, alpha
        if verbose ==1:
            return regression_slopes[:2], alpha
        if verbose == 0:
            return regression_slopes[0], alpha

    def regression_slopes2(self, cols, y_col, verbose=0):
        x, y = self.table[cols].values, self.table[y_col].values
        model = LinearRegression().fit(x,y)
        return model.coef_, model.intercept_

    def scatterPlot(self, x, y, subplot_dims=111):
        fig = plt.figure()
        ax=fig.add_subplot(subplot_dims)
        ax.scatter(self.table[x].values, self.table[y].values)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        plt.show()

    def new_col_from_func(self, colname, input_cols, function):
        self.table[colname] = function(self.table, input_cols)