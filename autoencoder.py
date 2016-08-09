import tensorflow as tf
import numpy as np
from os.path import join

class Autoencoder(object):

    def __init__(self, shape, sess):
        self._shape = shape #[input dim, frist layer dim, ..., output dim]
        self._hidden_layer_num = len(shape) - 2

        self._sess = sess
        # self._setup_variables()


    @property
    def shape(self):
        return self._shape

    @property
    def session(self):
        return self._sess

    @property
    def hidden_layer_num(self):
        return self._hidden_layer_num

    def _setup_variables(self):
        with tf.name_scope("autoencoder variables"):
            for i in xrange(self._hidden_layer_num+1):
                pass


def main():
    AE = Autoencoder([6,3,3,6], "tensor session")
    aaa = AE.session
    print aaa

if __name__ == "__main__":
    main()
