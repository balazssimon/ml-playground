import tensorflow as tf
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


batch_size = 128
hidden_layer_size = 1024


def prediction(dataset, weights1, biases1, weights2, biases2):
    return tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(dataset, weights1)+biases1), weights2)+biases2)


graph = tf.Graph()
with graph.as_default():
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
    outputs2 = tf.matmul(inputs2, weights2) + biases2

    # Layer 3 (output layer) computation:
    logits = outputs2
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = prediction(tf_valid_dataset, weights1, biases1, weights2, biases2)
    test_prediction = prediction(tf_test_dataset, weights1, biases1, weights2, biases2)

num_steps = 3001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


