import tensorflow as tf

from com.mnist_model import MnistModel
from com.test_model import TestModel


class TotalModel(tf.keras.Model):
    def __init__(self):
        super(TotalModel, self).__init__()

        self.mnist = MnistModel()

        self.test = TestModel()

    def call(self, x, y):
        p = self.mnist(x, is_training=True)

        loss = self.mnist.loss(p, y)

        q = self.test(x, is_training=True)
        q_loss = self.test.loss(q, y)

        return loss, q_loss
