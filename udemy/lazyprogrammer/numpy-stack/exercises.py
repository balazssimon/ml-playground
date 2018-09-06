import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Exercise 1

A = np.array([[0.3,0.6,0.1],[0.5,0.2,0.3],[0.4,0.1,0.5]])
v = np.array([1/3,1/3,1/3])

v_dist = []
v_next = v
for i in range(1,25):
    v_current = v_next
    v_next = v_next*A
    v_dist.append(np.linalg.norm(v_next-v_current))

plt.plot(v_dist)
plt.show


# Exercise 2

N = 1000
Ys = np.random.randn(10000,N)
Ysum = np.sum(Ys, axis=1)
plt.hist(Ysum, bins=100)
plt.show()

Ysum.mean()
Ysum.var()


# Exercise 3

df = pd.read_csv("train.csv")

digit = 7
D = df[df['label'] == digit]
D_mean = D.mean().values
im = D_mean[1:]
im = im.reshape(28,28)
plt.imshow(im, cmap="gray")
plt.show()

# Exercise 4

im90 = np.rot90(im,3)
plt.imshow(im90, cmap="gray")
plt.show()

# Exercise 5

def is_symmetric(M):
    return np.abs(M-M.T).sum() == 0

M1 = np.array([[1,2],[3,4]])
M2 = np.array([[1,2],[2,1]])

is_symmetric(M1)
is_symmetric(M2)

# Exercise 6

XOR = np.random.random((5000,2))*2-1
z = (np.sign(XOR[:,0]*XOR[:,1])+1)/2
z = np.expand_dims(z, axis=0).T
XOR = np.append(XOR, z, axis=1)
plt.scatter(XOR[:,0], XOR[:,1], c=XOR[:,2], cmap=plt.cm.RdBu, alpha=0.5)
plt.axis('equal')
plt.show()

# Exercise 7

N = 2000
C = np.random.random((N,2))
s = np.sign(C[:,1]-0.5)
r0 = np.random.randn(N)
r = (s-3)/2*10+r0
x = np.cos(C[:,0]*2*np.pi)*r
y = np.sin(C[:,0]*2*np.pi)*r
plt.scatter(x, y, c=s, cmap=plt.cm.RdBu, alpha=0.5)
plt.axis('equal')
plt.show()

# Exercise 8

N = 2000
t6 = np.random.randint(6, size=N)
s = np.mod(t6,2)
t = 0.5+np.random.random(N)*2+np.random.randn(N)*0.1
r = t
a = (t+t6)*np.pi/3
r0 = r+np.random.randn(N)*0.1
x = np.cos(a)*r0
y = np.sin(a)*r0
plt.scatter(x, y, c=s, cmap=plt.cm.RdBu, alpha=0.5)
plt.axis('equal')
plt.show()

df = pd.DataFrame(data={'x1': x, 'x2': y, 'y': s})
df.to_csv('spiral.csv', index=False)
