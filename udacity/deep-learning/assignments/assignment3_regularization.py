import tensorflow as tf
import numpy as np
from notmnist_common import load_prepared_dataset
from notmnist_common import accuracy


# loading the dataset:
pickle_file = 'notMNIST.pickle'
image_size = 28
num_labels = 10
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = load_prepared_dataset(pickle_file, image_size, num_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# should be 3001:
num_steps = 200001
report_steps = 1000

# should be 128:
batch_size = 128
hidden_layer_size = 2048

# problem 1:
# 1.0 and 0.1 are too large: underfitting
# 0.01 and 0.001 seem to be OK
beta = 0.00001

# problem 2:
overfit = False

# problem 3:
dropout = True

# problem 4:
starter_learning_rate = 0.5
decay_steps = num_steps
decay_rate = 0.96
learning_rate_decay = True


def prediction(dataset, weights1, biases1, weights2, biases2):
    return tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(dataset, weights1)+biases1), weights2)+biases2)


def train_network():
    ''' SGD, one hidden layer, L2-loss '''
    graph = tf.Graph()
    with graph.as_default():
        # Global step counter:
        global_step = tf.Variable(0)

        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Layer 1 (input layer) weights and biases:
        weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_layer_size]))
        biases1 = tf.Variable(tf.zeros([hidden_layer_size]))

        # Layer 1 (input layer) computation:
        outputs1 = tf.matmul(tf_train_dataset, weights1) + biases1

        # Layer 2 (hidden layer) weights and biases:
        weights2 = tf.Variable(tf.truncated_normal([hidden_layer_size, num_labels]))
        biases2 = tf.Variable(tf.zeros([num_labels]))

        # Layer 2 (hidden layer) computation:
        inputs2 = tf.nn.relu(outputs1)
        if dropout:
            inputs2 = tf.nn.dropout(inputs2, 0.5)
        outputs2 = tf.matmul(inputs2, weights2) + biases2

        # Layer 3 (output layer) computation:
        inputs3 = outputs2
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(inputs3, tf_train_labels)) + \
            beta*tf.nn.l2_loss(weights1) + beta*tf.nn.l2_loss(weights2)

        # Optimizer.
        if learning_rate_decay:
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate)
        else:
            learning_rate = starter_learning_rate
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # Predictions for the training, validation, and test data.
        logits = outputs2
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = prediction(tf_valid_dataset, weights1, biases1, weights2, biases2)
        test_prediction = prediction(tf_test_dataset, weights1, biases1, weights2, biases2)

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            if overfit:
                offset = np.random.choice(np.arange(10))
            else:
                offset = np.random.choice(np.arange(train_labels.shape[0] - batch_size))
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % report_steps == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("  Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("  Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
                print("  Learning rate: %f" % learning_rate.eval())
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

# problem1:
train_network()

''' problem2: overfitting
beta = 0.0
overfit = True
dropout = False
learning_rate_decay = false

Minibatch loss at step 0: 343.682800
Minibatch accuracy: 10.2%
Validation accuracy: 29.1%
Minibatch loss at step 500: 0.000000
Minibatch accuracy: 100.0%
Validation accuracy: 66.9%
Minibatch loss at step 1000: 0.000000
Minibatch accuracy: 100.0%
Validation accuracy: 66.9%
Minibatch loss at step 1500: 0.000000
Minibatch accuracy: 100.0%
Validation accuracy: 66.9%
Minibatch loss at step 2000: 0.000000
Minibatch accuracy: 100.0%
Validation accuracy: 66.9%
Minibatch loss at step 2500: 0.000000
Minibatch accuracy: 100.0%
Validation accuracy: 66.9%
Minibatch loss at step 3000: 0.000000
Minibatch accuracy: 100.0%
Validation accuracy: 66.9%
Test accuracy: 73.8%
'''



''' problem3: dropout
beta = 0.0
overfit = True
dropout = True
learning_rate_decay = false

Minibatch loss at step 0: 465.040466
Minibatch accuracy: 12.5%
Validation accuracy: 30.1%
Minibatch loss at step 500: 0.000000
Minibatch accuracy: 100.0%
Validation accuracy: 71.6%
Minibatch loss at step 1000: 0.000000
Minibatch accuracy: 100.0%
Validation accuracy: 71.5%
Minibatch loss at step 1500: 0.000000
Minibatch accuracy: 100.0%
Validation accuracy: 71.6%
Minibatch loss at step 2000: 0.000000
Minibatch accuracy: 100.0%
Validation accuracy: 71.2%
Minibatch loss at step 2500: 0.000000
Minibatch accuracy: 100.0%
Validation accuracy: 72.0%
Minibatch loss at step 3000: 0.000000
Minibatch accuracy: 100.0%
Validation accuracy: 71.9%
Test accuracy: 79.2%
'''


