import tensorflow as tf

"""
自定义Model要注意的点
 1. __init__若没有注册该layers，那么在后面应用梯度时会找不到model.trainable_variables
 2. 使用model.summary() 需要先指定input_shape :
            model.build(input_shape=(None, 28, 28, 1))
            model.summary()
"""


class MnistModel(tf.keras.Model):
    def __init__(self):
        super(MnistModel, self).__init__()

        # 注册layers

        # 卷积 3x3 32个卷积核
        self.conv2d_1 = tf.keras.layers.Conv2D(32, kernel_size=3, padding='SAME', activation=tf.nn.relu)

        # 池化 - 下采样 池化核尺寸为2 池化核步长为2
        self.maxpool_1 = tf.keras.layers.MaxPool2D(2, strides=2)  # 池化核尺寸,池化核步长

        # 卷积 3x3 64个卷积核
        self.conv2d_2 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='SAME', activation=tf.nn.relu)

        # 池化 - 下采样 池化核尺寸为2 池化核步长为2
        self.maxpool_2 = tf.keras.layers.MaxPool2D(2, strides=2)

        # 展开成一维数组
        self.flatten = tf.keras.layers.Flatten()

        # # Fully connected layer.(全连接层)
        self.fc_1 = tf.keras.layers.Dense(1024)

        # 丢弃一半
        self.dropout = tf.keras.layers.Dropout(rate=0.5)

        # 全连接层
        self.fc_2 = tf.keras.layers.Dense(10)

    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 28, 28, 1])

        # 卷积 [?,28,28,1] -> [?,28,28,32]
        x = self.conv2d_1(x)

        # 池化 [?,28,28,32] -> [?,14,14,32]
        x = self.maxpool_1(x)

        # 卷积 [?,14,14,32] -> [?,14,14,64]
        x = self.conv2d_2(x)

        # 池化 [?,14,14,64] -> [?,7,7,64]
        x = self.maxpool_2(x)

        # 展开成一维数组 [?,7,7,64] -> [?,3136]  7x7x64=3136
        x = self.flatten(x)

        # # Fully connected layer.(全连接层) -> [?,1024]
        x = self.fc_1(x)

        # 丢弃一半
        x = self.dropout(x, training=is_training)

        # 全连接层 [?,1024] -> [?,10]
        x = self.fc_2(x)

        return x
