import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data_poly.csv", header=None)

x = df[0].values
n = x.shape[0]
X0 = np.expand_dims(np.ones(n), axis=0).T
X1 = np.expand_dims(x, axis=0).T
X2 = np.expand_dims(x*x, axis=0).T
X = X0
X = np.append(X, X1, axis=1)
X = np.append(X, X2, axis=1)
y = df[1].values

w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))

y_hat = np.dot(X, w)

d1 = y-y_hat
d2 = y-y.mean()
SS_res = d1.dot(d1)
SS_tot = d2.dot(d2)
R2 = 1-SS_res/SS_tot


plt.scatter(x, y)
plt.plot(x, y_hat, c='red')
plt.show()
