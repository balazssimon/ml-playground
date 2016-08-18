import tensorflow as tf
import numpy as np
from notmnist_common import load_datasets
from notmnist_common import accuracy


# loading the dataset:
pickle_file = 'notMNIST.pickle'
image_size = 28
num_labels = 10
num_channels = 1  # grayscale
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = load_datasets(pickle_file)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

num_steps = 3001
report_steps = 50
batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

# problem 2:
dropout = True
starter_learning_rate = 0.05
decay_steps = num_steps
decay_rate = 0.96
learning_rate_decay = True

graph = tf.Graph()

with graph.as_default():
    keep_prob = tf.placeholder(tf.float32)
    # Global step counter:
    global_step = tf.Variable(0)

    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))


    # Model.
    def model(data):
        # problem 1: conv2d([1,2,2,1]) -> conv2d([1,1,1,1])+max_pool([1,2,2,1])
        conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        hidden1 = tf.nn.relu(conv1 + layer1_biases)
        pool1 = tf.nn.max_pool(hidden1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.conv2d(pool1, layer2_weights, [1, 1, 1, 1], padding='SAME')
        hidden2 = tf.nn.relu(conv2 + layer2_biases)
        pool2 = tf.nn.max_pool(hidden2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shape = pool2.get_shape().as_list()
        reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        # problem 2: dropout
        hidden = tf.nn.dropout(hidden, keep_prob)
        return tf.matmul(hidden, layer4_weights) + layer4_biases


    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # problem 2: learning rate decay
    if learning_rate_decay:
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate)
    else:
        learning_rate = starter_learning_rate
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))


with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    #offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    offset = np.random.choice(np.arange(train_labels.shape[0] - batch_size))
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    if dropout:
        dropout_keep_prob = 0.5
    else:
        dropout_keep_prob = 1.0
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob : dropout_keep_prob }
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % report_steps == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('  Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('  Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(feed_dict={ keep_prob: 1.0 }), valid_labels))
      print("  Learning rate: %f" % learning_rate.eval())
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(feed_dict={ keep_prob: 1.0 }), test_labels))
