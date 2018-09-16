import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel('mlr02.xls')

X = df.values

plt.scatter(X[:,1], X[:,0])
plt.show()
plt.scatter(X[:,2], X[:,0])
plt.show()
plt.scatter(X[:,1], X[:,2])
plt.show()

df['ones'] = 1 #np.random.rand(11)*100
Y = df['X1']
X = df[['X2','X3','ones']]
X2only = df[['X2','ones']]
X3only = df[['X3','ones']]

def get_r2(X,Y):
    w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
    Y_hat = np.dot(X, w)
    d1 = Y - Y_hat
    d2 = Y - Y.mean()
    SS_res = d1.dot(d1)
    SS_tot = d2.dot(d2)
    R2 = 1 - SS_res / SS_tot
    return R2

print("R2 for X2 only: ", get_r2(X2only, Y))
print("R2 for X3 only: ", get_r2(X3only, Y))
print("R2 for X2 and X3: ", get_r2(X, Y))
