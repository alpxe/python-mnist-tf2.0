from com.manager.Dataset import Dataset
from com.manager.Download import Download
import tensorflow as tf

from com.mnist_model import MnistModel


class Main:
    def __init__(self):
        Download().load()

        # 获取数据集
        dataset = Dataset().read()

        # 模型
        self.model = MnistModel()
        self.model.build(input_shape=(None, 28 * 28))

        # self.model.summary()

        # 读取所保存的模型权重 (有文件后可以去掉注释)
        # self.model.load_weights('checkpoints/my_checkpoint')

        # 梯度下降优化器
        self.optimizer = tf.optimizers.Adam(0.0001)

        # image shape=(10,784)
        for step, (label, image) in enumerate(dataset):
            loss = self.run_optimization(image, label)
            print("loss: {}".format(loss))

            # if loss < 0.1:
            #     # 保存模型权重(覆盖)
            #     self.model.save_weights('checkpoints/my_checkpoint')
            #     break

    # Optimization process.
    def run_optimization(self, x, y):
        with tf.GradientTape() as tape:
            # 向前传递 获得预测值
            pred = self.model(x, is_training=True)
            softmax = tf.nn.softmax(pred)

            # 有样本值(真实值) 与 预测值 => 计算loss
            # H(x)=-∑P(xᵢ)·log₂[P(xᵢ)]  api:  tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=softmax, axis=1)
            loss = -tf.reduce_sum(y * (tf.math.log(softmax) / tf.math.log(2.)), axis=1)
            loss = tf.reduce_mean(loss)

        # 更新训练的变量 trainable_variables = conv_net.trainable_variables
        trainable_variables = self.model.trainable_variables

        # Compute gradient 梯度计算
        gradient = tape.gradient(loss, trainable_variables)

        # 梯度更新
        self.optimizer.apply_gradients(zip(gradient, trainable_variables))

        return loss.numpy()

    pass


pass

if __name__ == "__main__":
    print(tf.__version__)
    Main()
