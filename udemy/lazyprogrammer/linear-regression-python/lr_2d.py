import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

df = pd.read_csv("data_2d.csv", header=None)

X = df[[0,1]].values
y = df[2].values

w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))

y_hat = np.dot(X, w)

d1 = y-y_hat
d2 = y-y.mean()
SS_res = d1.dot(d1)
SS_tot = d2.dot(d2)
R2 = 1-SS_res/SS_tot


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], y)
plt.show()
