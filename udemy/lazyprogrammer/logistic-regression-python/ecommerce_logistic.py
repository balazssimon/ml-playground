import numpy as np
from ecommerce_data import get_binary_data
import matplotlib.pyplot as plt

Xtrain, Ytrain, Xtest, Ytest = get_binary_data()

D = Xtrain.shape[1]

Ntrain = Xtrain.shape[0]
ones = np.array([[1]*Ntrain]).T
Xtrain = np.concatenate((ones, Xtrain), axis=1)

Ntest = Xtest.shape[0]
ones = np.array([[1]*Ntest]).T
Xtest = np.concatenate((ones, Xtest), axis=1)

# randomly initialize weights
W = np.random.randn(D+1)

# make predictions
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

# cross entropy
def cross_entropy(Y, P):
    return -np.mean(Y*np.log(P+0.0000001) + (1 - Y)*np.log(1 - P+0.0000001))

# calculate the accuracy
def classification_rate(Y, P):
    return np.mean(Y == P)

# gradient descent
def gradient_descent(L1=False, L2=False):
    train_costs = []
    test_costs = []
    y = Ytrain
    w = W
    l1 = 0.01
    l2 = 0.01
    learning_rate = 0.001
    for i in range(1000):
        a = Xtrain.dot(w)
        y = sigmoid(a)

        ctrain = cross_entropy(y, Ytrain)
        train_costs.append(ctrain)
        ctest = cross_entropy(sigmoid(Xtest.dot(w)), Ytest)
        test_costs.append(ctest)

        # gradient descent
        cost = learning_rate * Xtrain.T.dot(y - Ytrain)
        if L1:
            cost += l1*np.sign(w)
        if L2:
            cost += l2*w
        w -= cost

        if i % 100 == 0:
            print(i, ctrain, ctest)

    Ptrain = np.round(sigmoid(Xtrain.dot(w)))
    Ptest = np.round(sigmoid(Xtest.dot(w)))
    print("Final w:", w)
    print("Train score:", classification_rate(Ptrain, Ytrain))
    print("Test score:", classification_rate(Ptest, Ytest))

    legend1, = plt.plot(train_costs, label='train cost')
    legend2, = plt.plot(test_costs, label='test cost')
    plt.legend([legend1, legend2])
    plt.show()

#gradient_descent()
#gradient_descent(L2=True)
#gradient_descent(L1=True)
gradient_descent(L1=True, L2=True)
