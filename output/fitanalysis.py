import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fitsdf = (pd.read_csv('fittdes.csv'))


print('overall', fitsdf.mean())

for i in fitsdf.columns:
    x = fitsdf[i].unique()
    if len(x) < 5:
        for j in x:
            print('%s = %s' %(i, j), fitsdf.loc[fitsdf[i] == j].mean())
            # print('%s = %s' %(i, j), round(fitsdf.loc[fitsdf[i] == j]['powerlaw'].mean() / fitsdf.loc[fitsdf[i] == j].mean(), 2))