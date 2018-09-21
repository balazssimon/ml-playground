import numpy as np
import pandas as pd
from scipy import stats

data = pd.read_csv("advertisement_clicks.csv")
X1 = data[data['advertisement_id'] == 'A']
X2 = data[data['advertisement_id'] == 'B']

A0 = X1[X1['action'] == 0].shape[0]
A1 = X1[X1['action'] == 1].shape[0]
B0 = X2[X2['action'] == 0].shape[0]
B1 = X2[X2['action'] == 1].shape[0]

T = [[A0, A1], [B0, B1]]

chi2, p, dof, ex = stats.chi2_contingency(T, correction=False)
