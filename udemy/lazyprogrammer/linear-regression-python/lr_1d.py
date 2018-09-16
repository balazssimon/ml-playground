import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data_1d.csv", header=None)

x = df[0].values
y = df[1].values

x_mean = x.mean()
x2_mean = (x*x).mean()
x_mean2 = x_mean*x_mean
y_mean = y.mean()
xy_mean = (x*y).mean()

a = (xy_mean-x_mean*y_mean)/(x2_mean-x_mean2)
b = (y_mean*x2_mean-x_mean*xy_mean)/(x2_mean-x_mean2)
y_hat = a*x + b

plt.scatter(x, y)
plt.plot(x, y_hat, c='red')
plt.show()

d1 = y-y_hat
d2 = y-y.mean()
SS_res = d1.dot(d1)
SS_tot = d2.dot(d2)
R2 = 1-SS_res/SS_tot
