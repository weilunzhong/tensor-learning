import tensorflow as tf
import numpy as np
import input_data


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder('float', [None, 784])
Y = tf.placeholder('float', [None, 10])

w = tf.Variable(tf.random_normal([784, 10], stddev=0.01))
y_model = tf.mul(X, w)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_model, Y))
training = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
prediction = tf.argmax(y_model, 1)

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            sess.run(training, feed_dict={X: trx[start:end], Y: trY[start:end]})
            print(i, np.mean(np.argmax(teY, axis=1) ==
                                         sess.run(predict_op, feed_dict={X: teX, Y: teY})))
