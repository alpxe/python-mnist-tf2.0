from com.manager.Dataset import Dataset
from com.manager.Download import Download
import tensorflow as tf

from com.total_model import TotalModel


class Main:
    def __init__(self):
        Download().load()

        # 获取数据集
        dataset = Dataset().read()

        # 模型
        self.model = TotalModel()
        # self.model.build(input_shape=(None, 28 * 28))

        # self.model.summary()

        # 读取所保存的模型权重 (有文件后可以去掉注释)
        # self.model.load_weights('checkpoints/my_checkpoint')

        # 梯度下降优化器
        self.optimizer = tf.optimizers.Adam(0.0001)

        # image shape=(10,784)
        for step, (label, image) in enumerate(dataset):
            loss = self.run_optimization(image, label)
            print("loss: {}".format(loss))

            # break
            # if loss < 0.1:
            #     # 保存模型权重(覆盖)
            #     self.model.save_weights('checkpoints/my_checkpoint')
            #     break
        self.model.summary()

    # Optimization process.
    def run_optimization(self, x, y):
        with tf.GradientTape() as tape:
            # 向前传递 获得预测值
            loss, q_loss = self.model(x, y)
            totaloss = loss + q_loss

        # 更新训练的变量 trainable_variables = conv_net.trainable_variables
        trainable_variables = self.model.trainable_variables

        # Compute gradient 梯度计算
        gradient = tape.gradient(totaloss, trainable_variables)

        # 梯度更新
        self.optimizer.apply_gradients(zip(gradient, trainable_variables))

        return totaloss.numpy()

    pass


pass

if __name__ == "__main__":
    print(tf.__version__)
    Main()
