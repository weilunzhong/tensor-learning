import tensorflow as tf
import input_data

def regression_approach():
    # None for batch size and 784 is the image size flattened, 10 digits for classification
    x = tf.placeholder("float", shape=[None, 784], name="x_input")
    y_ = tf.placeholder("float", shape=[None, 10], name="y_input")

    # init of weights and bias
    W = tf.Variable(tf.zeros([784,10]), name="weights")
    b = tf.Variable(tf.zeros([10]), name="bias")

    # init vairables in a session
    # sess.run(tf.initialize_all_variables())

    # predictions
    with tf.name_scope("prediction") as scope:
        y = tf.nn.softmax(tf.matmul(x, W) + b)

    w_hist = tf.histogram_summary("weights", W)
    b_hist = tf.histogram_summary("biases", b)
    y_hist = tf.histogram_summary("y", y)

    # cost func set to be cross entropy
    with tf.name_scope("cost_func") as scope:
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        ce_sum = tf.scalar_summary("cross_entropy", cross_entropy)

    # training happens after all is defined and steepest gradient descent is used
    with tf.name_scope("train") as scope:
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

	# evaluate model
	with tf.name_scope("test") as scope:
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            accuracy_summary = tf.scalar_summary("accuracy", accuracy)

	# merge all the summaries
	merged = tf.merge_all_summaries()

	# Launch the graph
	with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            writer = tf.train.SummaryWriter('/home/vionlabs/Documents/vionlabs_weilun/machine_learning/tensorflow_testing/graph', graph_def=sess.graph_def)
            # tf.initialize_all_variables().run()

            for i in range(1000):
                if i % 10 == 0:  # Record summary data, and the accuracy
                    feed = {x: mnist.test.images, y_: mnist.test.labels}
                    result = sess.run([merged, accuracy], feed_dict=feed)
                    summary_str = result[0]
                    acc = result[1]
                    writer.add_summary(summary_str, i)
                    print("Accuracy at step %s: %s" % (i, acc))
                else:
                    batch_xs, batch_ys = mnist.train.next_batch(100)
                    feed = {x: batch_xs, y_: batch_ys}
                    sess.run(train_step, feed_dict=feed)


            print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})

if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    regression_approach()
