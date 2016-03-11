import tensorflow as tf
import numpy as np

# variable creation func
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name="weights")

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name="bias")

# func that defines the layer
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name="conv2d")

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="pool")

input_data = np.ones([2, 784])
x = tf.placeholder("float", shape=[None, 784], name="x_input")
x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, prob)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
y = h_fc1_drop.eval(feed_dict={x: input_data, prob: 0.1})
# y = b_conv1.eval()
print y.shape
