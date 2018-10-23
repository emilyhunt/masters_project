import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sns

from scipy.stats import norm
from sklearn.model_selection import train_test_split

# Set seeds
the_meaning_of_life = 42
np.random.seed(the_meaning_of_life)
tf.set_random_seed(the_meaning_of_life)

# Create a toy dataset
def build_toy_dataset(N):
    y_data = np.random.uniform(-10.5, 10.5, N)
    r_data = np.random.normal(size=N)  # random noise
    x_data = np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0
    x_data = x_data.reshape((N, 1))
    y_data = y_data.reshape((N, 1))
    return train_test_split(x_data, y_data, random_state=42)


points = 5000  # number of data points
features = 1  # number of input features
mixture_components = 20  # number of mixture components
layer_size = 15

x_train, x_test, y_train, y_test = build_toy_dataset(points)
print("Size of features in training data: {}".format(x_train.shape))
print("Size of output in training data: {}".format(y_train.shape))
print("Size of features in test data: {}".format(x_test.shape))
print("Size of output in test data: {}".format(y_test.shape))
plt.plot(x_train, y_train, 'or', mew=0, ms=3, alpha=0.2)
plt.title('Training data')
plt.show()


# Setup our tensorflow model
# Define some placeholders for data
x_placeholder = tf.placeholder(tf.float32, [None, features])
y_placeholder = tf.placeholder(tf.float32, [None, features])

# Setup our layers
hidden_layer_1 = tf.layers.dense(x_placeholder, layer_size, activation=tf.nn.relu)
hidden_layer_2 = tf.layers.dense(hidden_layer_1, layer_size, activation=tf.nn.relu)
means = tf.layers.dense(hidden_layer_2, mixture_components, activation=tf.nn.softmax)
std_deviations = tf.layers.dense(hidden_layer_2, mixture_components, activation=tf.exp)
mixture_weights = tf.layers.dense(hidden_layer_2, mixture_components, activation=tf.nn.softmax)

# Define a normal distribution
oneDivSqrtTwoPI = 1 / np.sqrt(2*np.pi)


def tf_mixture_normal(a_point, my_means, my_std_deviations):
    """Normal distribution implemented in tensorflow notation."""
    result = tf.subtract(a_point, my_means)
    result = tf.multiply(result, tf.reciprocal(my_std_deviations))
    result = -tf.square(result)/2
    return tf.multiply(tf.exp(result), tf.reciprocal(my_std_deviations)) * oneDivSqrtTwoPI


def get_loss_func(a_point, my_weights, my_means, my_std_deviations):
    """Lossfunc defined in tensorflow notation."""
    # Calculate normal distribution mixture and normalise
    result = tf_mixture_normal(a_point, my_means, my_std_deviations)
    result = tf.multiply(result, my_weights)

    # Sum the result and take the mean negative log
    # todo: mean log is sensitive to outliers
    # todo: add some kind of L2 regularisation, penalising poor weight choices
    result = tf.reduce_sum(result, 1, keepdims=True)
    result = -tf.log(result)
    return tf.reduce_mean(result)


# Setup de-facto pointers to loss & training functions
loss_func = get_loss_func(y_placeholder, mixture_weights, means, std_deviations)
train_func = tf.train.AdamOptimizer().minimize(loss_func)

# Session parameters
n_epochs = 2000
loss = np.zeros(n_epochs)

# Run the session a few times
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(n_epochs):
    # Setup a feed dictionary of data
    cookies = {x_placeholder: x_train, y_placeholder: y_train}

    # Feed the session some cookies (...aka data)
    sess.run(train_func, feed_dict=cookies)
    loss[i] = sess.run(loss_func, feed_dict=cookies)

    # Update the user on progress
    print('Epoch = {}, loss = {}'.format(i, loss[i]))

plt.figure()
plt.plot(np.arange(0, n_epochs), loss, 'r-')
plt.title('Loss function evolution')
plt.show()


# todo: add random distribution sampling