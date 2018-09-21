import numpy as np
import pandas as pd
from scipy import stats

data = pd.read_csv("advertisement_clicks.csv")
X1 = data[data['advertisement_id'] == 'A']
X2 = data[data['advertisement_id'] == 'B']

n1 = X1.shape[0]
n2 = X2.shape[0]
N = n1

x1 = X1['action'].values
x2 = X2['action'].values

t, p = stats.ttest_ind(x1, x2)
print("t:\t", t, "p:\t", 2*p) # two-sided test p-value

t, p = stats.ttest_ind(x1, x2, equal_var=False)
print("Welch's t-test:")
print("t:\t", t, "p:\t", p)
