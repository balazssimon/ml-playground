# Boltzmann Machines

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv) # weights
        self.a = torch.randn(1, nh)  # bias for the visible layer
        self.b = torch.randn(1, nv)  # bias for the hidden layer
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t()) # x * W^T
        activation = wx + self.a.expand_as(wx) # the dimension of the bias a is expanded to the dimension of wx
        p_h_given_v = torch.sigmoid(activation) # convert the activation to range 0..1 (probability that the given hidden node is activated)
        return p_h_given_v, torch.bernoulli(p_h_given_v) # activation probabilities + converted to 0 (below 70% probability) and 1 (above 70% probability)
    def sample_v(self, y):
        wy = torch.mm(y, self.W) # y * W
        activation = wy + self.b.expand_as(wy) # the dimension of the bias b is expanded to the dimension of wx
        p_v_given_h = torch.sigmoid(activation) # convert the activation to range 0..1 (probability that the given visible node is activated)
        return p_v_given_h, torch.bernoulli(p_v_given_h) # activation probabilities + converted to 0 (below 70% probability) and 1 (above 70% probability)
    def train(self, v0, vk, ph0, phk):
        # v_0: all the ratings by one user
        # v_k: visible node after k iterations (samplings)
        # P(nh==1|v_0): probability that the hidden nodes are activated given the visible nodes have value v_0
        # P(nh==1|v_k): probability that the hidden nodes are activated given the visible nodes have value v_k
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t() # v_0^T * P(nh==1|v_0) - v_k^T * P(nh==1|v_k)
        self.b += torch.sum((v0 - vk), 0) # sum: v_0 - v_k
        self.a += torch.sum((ph0 - phk), 0) # sum: P(nh==1|v_0) - P(nh==1|v_k) 
nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0] # freezing the -1 ratings (no ratings): restoring them to their original values
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0])) # Average distance
        # train_loss += np.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2)) # RMSE
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0])) # Average distance
        # test_loss += np.sqrt(torch.mean((vt[vt>=0] - v[vt>=0])**2)) # RMSE
        s += 1.
print('test loss: '+str(test_loss/s))