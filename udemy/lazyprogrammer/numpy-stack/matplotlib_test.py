import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x = np.linspace(0, 10, 10) # interval 0..10 in 10 elements
y = np.sin(x)

plt.plot(x,y)
plt.xlabel("Time")
plt.ylabel("Some function of time")
plt.title("My cool chart")
plt.show()


x = np.linspace(0, 10, 100) # interval 0..10 in 10 elements
y = np.sin(x)
plt.plot(x,y)
plt.show()


A = pd.read_csv("data_1d.csv", header=None).values
x = A[:,0]
y = A[:,1]

plt.scatter(x, y)
plt.show()

x_line = np.linspace(0, 100, 100)
y_line = 2*x_line+1

plt.scatter(x, y)
plt.plot(x_line, y_line)
plt.show()

# histogram:
plt.hist(x)
plt.show()

R = np.random.random(10000)
plt.hist(R)
plt.show()

plt.hist(R, bins=20)
plt.show()

y_actual = 2*x+1
residuals = y - y_actual
plt.hist(residuals)
plt.show()


df = pd.read_csv("train.csv")
df.shape
M = df.values
im = M[0,1:]
im.shape
im = im.reshape(28,28)
im.shape
plt.imshow(im)
plt.show()
M[0,0]
plt.imshow(im, cmap="gray")
plt.show()
plt.imshow(255-im, cmap="gray")
plt.show()
