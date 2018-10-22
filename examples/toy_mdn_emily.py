import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sns

from scipy.stats import norm
from sklearn.model_selection import train_test_split

# Set seeds
np.random.seed(42)
tf.set_random_seed(42)


# Create a toy dataset
def build_toy_dataset(N):
    y_data = np.random.uniform(-10.5, 10.5, N)
    r_data = np.random.normal(size=N)  # random noise
    x_data = np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0
    x_data = x_data.reshape((N, 1))
    return train_test_split(x_data, y_data, random_state=42)


points = 5000  # number of data points
features = 1  # number of features
mixture_components = 20  # number of mixture components
layers = 15

X_train, X_test, y_train, y_test = build_toy_dataset(points)
print("Size of features in training data: {}".format(X_train.shape))
print("Size of output in training data: {}".format(y_train.shape))
print("Size of features in test data: {}".format(X_test.shape))
print("Size of output in test data: {}".format(y_test.shape))
plt.plot(X_train, y_train, 'or', mew=0, ms=3, alpha=0.2)
plt.show()


# Setup our tensorflow model
# Define some placeholders for data
x_placeholder = tf.placeholder(tf.float32, [None, features])
y_placeholder = tf.placeholder(tf.float32, [None])

# Setup our layers
hidden_layer_1 = tf.layers.dense(x_placeholder, layers, activation=tf.nn.relu)
hidden_layer_2 = tf.layers.dense(hidden_layer_1, layers, activation=tf.nn.relu)
means = tf.layers.dense(hidden_layer_2, mixture_components, activation=None)
std_devs = tf.layers.dense(hidden_layer_2, mixture_components, activation=tf.exp)
mixture_weights = tf.layers.dense(hidden_layer_1, mixture_components, activation=None)




