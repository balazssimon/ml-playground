# AutoEncoders

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

# Stacked Auto Encoder class: creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__() # calling the super constructor
        self.fc1 = nn.Linear(nb_movies, 20) # fully connected NN between the input nodes and the first hidden layer
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
sae = SAE()
criterion = nn.MSELoss() # loss function: Mean Squared Error
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) # optimizer: RMS prop (there are others, e.g. Adam)

# Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0) # a batch of a single vector (pythorch needs a batch of vectors, not a single vector)
        target = input.clone() # copy the input
        if torch.sum(target.data > 0) > 0: # contains at least one rating
            output = sae(input) # calling the sae object on the input is equivalent to prediction
            target.require_grad = False # saves some coputations and memory: we don't need the gradient of the target variable for training
            output[target == 0] = 0 # set the output to zero where the target is zero: these won't contribute to the error
            loss = criterion(output, target) # calculate the error using the loss function
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) # average of the error for the movies we are rating: penalty for users with less ratings (the 1e-10 is to make sure that the denominator is always non-zero)
            loss.backward() # backpropagation: decrease the weights with the gradient (loss.forward(): increase)
            train_loss += np.sqrt(loss.item()*mean_corrector) # adjusting the loss with the mean_corrector
            s += 1. # increase the number of users who have ratings
            optimizer.step() # run the optimizer to update the weights: calculates the amount with which the weights are updated (in the direction defined by the loss.backward()/forward())
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0) # the training set contains the movies the user has rated before
    target = Variable(test_set[id_user]).unsqueeze(0) # the test set contains the real ratings for the movies we want to rate before the user sees them
    if torch.sum(target.data > 0) > 0:
        output = sae(input) # calling the sae object on the input is equivalent to prediction
        target.require_grad = False # saves some coputations and memory: we don't need the gradient of the target variable for prediction
        output[target == 0] = 0 # set the output to zero where the target is zero: these won't contribute to the error
        loss = criterion(output, target) # calculate the error using the loss function
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) # average of the error for the movies we are rating: penalty for users with less ratings (the 1e-10 is to make sure that the denominator is always non-zero)
        test_loss += np.sqrt(loss.item()*mean_corrector) # adjusting the loss with the mean_corrector
        s += 1. # increase the number of users who have ratings
print('test loss: '+str(test_loss/s))