import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.randn(N,D)
# center the first 50 points at (-2,-2)
X[:50,:] = X[:50,:]-2*np.ones((50,D))
# center the last 50 points at (+2,+2)
X[50:,:] = X[50:,:]+2*np.ones((50,D))
T = np.array([0]*50+[1]*50)
ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis=1)

w = np.random.randn(D+1)

a = Xb.dot(w)

def sigmoid(z):
    return 1/(1+np.exp(-z))

z = sigmoid(a)
print(z)

# cross entropy
def cross_entropy(Y, P):
    return -np.mean(Y*np.log(P+0.0000001) + (1 - Y)*np.log(1 - P+0.0000001))

print("Cross-entropy:", cross_entropy(T, z))

plt.scatter(X[:,0], X[:,1], c=T, s=100, alpha=0.5)
x_axis = np.linspace(-6, 6, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)
plt.show()

# train loop
Xtrain = Xb
Ytrain = z
train_costs = []
learning_rate = 0.001
for i in range(100):
    a = Xb.dot(w)
    Ytrain = sigmoid(a)

    ctrain = cross_entropy(T, Ytrain)
    train_costs.append(ctrain)

    # gradient descent
    w -= learning_rate * Xtrain.T.dot(Ytrain - T)
    if i % 10 == 0:
        print(i, ctrain)

print(Ytrain)
print("Final train cross_entropy:", cross_entropy(Ytrain, T))

legend1, = plt.plot(train_costs, label='train cost')
plt.legend([legend1])
plt.show()

