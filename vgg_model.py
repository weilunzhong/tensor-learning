import tensorflow as tf
from datetime import datetime
import input_data
import time
import numpy as np
from dataset import get_cifar10

DEFAULT_PADDING = 'SAME'
batch_size = 128

class VggModel(object):

    def __init__(self, model_path=None):
        if model_path:
            self.path = model_path + 'vgg.npy'

    @staticmethod
    def set_weight_variable(shape, stddev=0.1):
        initial = tf.truncated_normal(shape, stddev)
        return tf.Variable(initial, name='weights')

    @staticmethod
    def set_bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)

    def _conv(self, input, name, k_h, k_w,
             o_c, s_w, s_h, relu=True, padding=DEFAULT_PADDING):
        """
        k_w, k_h are kernal width and height
        o_c, i_c are in/output channels
        s_w, s_h are convolution stride width and height
        """
        i_c = input.get_shape().as_list()[-1]
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.name_scope(name) as scope:
            # weights = self.set_weight_variable(shape=[k_h, k_w, i_c, o_c])
            init_weights = tf.truncated_normal(shape=[k_h, k_w, i_c, o_c], stddev=0.1)
            weights = tf.Variable(init_weights, trainable=True, name='weights')
            # bias = self.set_bias_variable(shape=[o_c])
            init_bias = tf.constant(0.1, shape=[o_c])
            bias = tf.Variable(init_bias, name='bias', trainable=True)
            conv = convolve(input, weights)
            output = tf.nn.bias_add(conv, bias)
            print output.get_shape().as_list(), name + " out shape"
            if relu:
                return tf.nn.relu(output)
            else:
                return output

    def _fully_connected(self, input, name, num_out, relu=True):
        """
        num_out is the number of output channels after the fc layer
        """
        input_dims = input.get_shape().as_list()[1:]
        with tf.variable_scope(name) as scope:
            dims = reduce(lambda i, j: i*j, input_dims)
            input_flat = tf.reshape(input, shape=[-1, dims])
            # weights = self.set_weight_variable(shape=[dims, num_out])
            init_weights = tf.truncated_normal(shape=[dims, num_out], stddev=0.1)
            weights = tf.Variable(init_weights, trainable=True, name='weights')
            # bias = self.set_bias_variable(shape=[num_out])
            init_bias = tf.constant(0.1, shape=[num_out])
            bias = tf.Variable(init_bias, name='bias', trainable=True)
            output = tf.nn.bias_add(tf.matmul(input_flat, weights), bias)
            if relu:
                output = tf.nn.relu(output)
            print output.get_shape().as_list(), name + " out shape"
            return output

    def _maxpool(self, input, name, k_h, k_w, s_h, s_w, padding=DEFAULT_PADDING):
        """
        k_h, k_w, s_h s_w are the same as conv definition
        """
        if padding not in ("SAME", "VALID"):
            raise TypeError("padding must be either SAME or VALID")
        with tf.variable_scope(name) as scope:
            output = tf.nn.max_pool(input,
                                  ksize=[1, k_h, k_w,1],
                                  strides=[1, s_h, s_w, 1],
                                  padding=padding,
                                  name=name)
            print output.get_shape().as_list(), name + " out shape"
            return output

    def _loss(self, logits, labels):
        # labels = tf.expand_dims(labels, 1)
        # indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
        # concated = tf.concat(1, [indices, labels])
        # onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, 10]), 1.0, 1.0)
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(labels, tf.float32), name='entropy')
        # loss = tf.reduce_mean(cross_entropy, name='entropy_mean')


        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='entropy')
        loss = tf.reduce_mean(cross_entropy, name='entropy_mean')
        # print labels, logits
        # cross_entropy = -tf.reduce_sum(tf.cast(labels, 'float')*tf.log(logits))
        return loss

    def inference(self, input, dropout_prob):
        # assert input.shape = (n, 224, 224, 3)

        # layer 1, output 112x112x64
        global fc7_drop, fc8
        conv1_1 = self._conv(input, name='conv1_1', k_h=3, k_w=3, o_c=64, s_w=1, s_h=1)
        conv1_2 = self._conv(conv1_1, name='conv1_2', k_h=3, k_w=3, o_c=64, s_w=1, s_h=1)
        pool1 = self._maxpool(conv1_2, name='pool1', k_h=2, k_w=2, s_h=2, s_w=2)

        # layer 2, output 56x56x128
        conv2_1 = self._conv(pool1, name='conv2_1', k_h=3, k_w=3, o_c=128, s_w=1, s_h=1)
        conv2_2 = self._conv(conv2_1, name='conv2_2', k_h=3, k_w=3, o_c=128, s_w=1, s_h=1)
        pool2 = self._maxpool(conv2_2, name='pool2', k_h=2, k_w=2, s_h=2, s_w=2)

        # # layer 3, output 28x28x256
        # conv3_1 = self._conv(pool2, name='conv3_1', k_h=3, k_w=3, o_c=256, s_w=1, s_h=1)
        # conv3_2 = self._conv(conv3_1, name='conv3_2', k_h=3, k_w=3, o_c=256, s_w=1, s_h=1)
        # pool3 = self._maxpool(conv3_2, name='pool3', k_h=2, k_w=2, s_h=2, s_w=2)

        # # layer 4, output 14x14x512
        # conv4_1 = self._conv(pool3, name='conv4_1', k_h=3, k_w=3, o_c=512, s_w=1, s_h=1)
        # conv4_2 = self._conv(conv4_1, name='conv4_2', k_h=3, k_w=3, o_c=512, s_w=1, s_h=1)
        # conv4_3 = self._conv(conv4_2, name='conv4_3', k_h=3, k_w=3, o_c=512, s_w=1, s_h=1)
        # pool4 = self._maxpool(conv4_2, name='pool4', k_h=2, k_w=2, s_h=2, s_w=2)

        # # layer 5, output 7x7x512
        # conv5_1 = self._conv(pool4, name='conv5_1', k_h=3, k_w=3, o_c=512, s_w=1, s_h=1)
        # conv5_2 = self._conv(conv5_1, name='conv5_2', k_h=3, k_w=3, o_c=512, s_w=1, s_h=1)
        # conv5_3 = self._conv(conv5_2, name='conv5_3', k_h=3, k_w=3, o_c=512, s_w=1, s_h=1)
        # pool5 = self._maxpool(conv5_2, name='pool5', k_h=2, k_w=2, s_h=2, s_w=2)

        # # fully connected
        # fc6 = self._fully_connected(pool5, name='fc6', num_out=4096)
        # fc6_drop = tf.nn.dropout(fc6, dropout_prob, name="fc6_drop")

        fc7 = self._fully_connected(pool2, name='fc7', num_out=1024)
        fc7_drop = tf.nn.dropout(fc7, dropout_prob, name="fc6_drop")

        fc8 = self._fully_connected(fc7_drop, name='fc8', num_out=10)
        predictions = tf.nn.softmax(fc8)

        return predictions

    def evaluate(self, predictions, labels):
        """
        Evaluate the predictions wrt labels

        Args:
            logits: Logits tensor, float - [batch_size, num_class]
            labels: Labels tensor, int32 - [batch_size], value in range(0, num_class)
        Returns:
            A scalar in32 tensor with the number of right prediction
            A ratio of correct prediction
        """
        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.cast(labels, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        return accuracy, correct_prediction

    def training(self, max_step=100000):

        with tf.Graph().as_default():
            in_imgs = tf.placeholder("float", [batch_size, 32, 32, 3])
            #images = tf.image.resize_images(in_imgs, 64, 64)
            labels = tf.placeholder("int32", [batch_size])
            dropout_prob = tf.placeholder("float")

            predictions = self.inference(in_imgs, dropout_prob)

            objective = self._loss(predictions, labels)
            _, total_correct = self.evaluate(predictions, labels)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            lr = tf.train.exponential_decay(
                    learning_rate=0.0001,
                    global_step=global_step,
                    decay_steps=500,
                    decay_rate=0.1,
                    staircase=True)
            optimizer = tf.train.MomentumOptimizer(lr, 0.9)
            train_step = optimizer.minimize(objective, global_step=global_step)

            # ema = tf.train.ExponentialMovingAverage(0.999)
            # maintain_averages_op = ema.apply([objective])

            # # write summary
            # tf.scalar_summary('loss function', objective)
            # # tf.scalar_summary('accuracy', accuracy)
            # tf.scalar_summary('avg loss function', ema.average(objective))

            # # saver
            # saver = tf.train.Saver(tf.all_variables())

            # summary_op = tf.merge_all_summaries()

            initializer = tf.initialize_all_variables()

            with tf.Session() as sess:
                sess.run(initializer)
                writer = tf.train.SummaryWriter('training logs', graph_def=sess.graph_def)
                trn, tst = get_cifar10(batch_size)
                for step in range(max_step):
                    batch_data = trn.next()
                    X = np.vstack(batch_data[0]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                    # X = X/255.
                    Y = np.array(batch_data[1])
                    # Y_LABEL = np.zeros((10))
                    # Y_LABEL[Y[0]] = 1
                    # Y_LABEL = np.expand_dims(Y_LABEL, axis=0)
                    # batch = mnist.train.next_batch(batch_size)

                    start_time = time.time()
                    result = sess.run(
                            [train_step, objective, fc8, predictions, lr],
                            feed_dict={
                                in_imgs: X,
                                labels: Y,
                                dropout_prob: 0.5
                                }
                            )
                    duration = time.time()-start_time


                    if np.isnan(result[1]):
                        print 'gradient exploded and result in nan'
                        return

                    if step % 100 == 0:
                        examples_per_sec = batch_size/duration
                        sec_per_batch = float(duration)
                        format_str = '%s: step %d, loss = %.4f (%.1f examples/sec; %.3f sec/batch)'
                        print(format_str % (datetime.now(), step, result[1], examples_per_sec, sec_per_batch))
                        print "learning rate is {0}".format(result[4])
                        # print result[2], 'fc8'
                        # print result[3], 'predictions'
                        # print batch[1]

                    # if step % 100 == 0:
                    #     writer.add_summary(result[2], step)

                    if step % 500 == 0:
                        print("%s: step %d, evaluating test set" % (datetime.now(), step))
                        correct_count = 0
                        tst_data = tst[0][:1280,:]
                        tst_label = tst[1][:1280]
                        num_tst_examples = tst_data.shape[0]
                        for tst_idx in range(0, num_tst_examples, batch_size):
                            X_tst = tst_data[tst_idx:np.min([tst_idx+batch_size, num_tst_examples]), :]
                            X_tst = X_tst.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                            Y_tst = tst_label[tst_idx:np.min([tst_idx+batch_size, num_tst_examples])]
                            correct_count += total_correct.eval({
                                in_imgs: X_tst,
                                labels: Y_tst,
                                dropout_prob: 1.0
                            })
                        accuracy = float(sum(correct_count))/num_tst_examples
                        print("%s tst accuracy = %.3f" % (datetime.now(), accuracy))
                        # if accuracy > 0.9:
                        #     checkpoint_path = saver.save(sess, "data/model.ckpt")
                        #     print("saving model %s" % checkpoint_path)


if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    vgg = VggModel()
    vgg.training()
