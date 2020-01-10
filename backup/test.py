from com.manager.Dataset import Dataset
from com.manager.Download import Download
import tensorflow as tf
from tensorflow import keras
import numpy as np


class Main:
    def __init__(self):
        Download().load()
        dataset = Dataset().read()

        # train_iter = iter(dataset)
        # sample = next(train_iter)
        # print("image_batch_size: {0}".format(sample["image"].shape))
        # print("label_batch_size: {0}".format(sample["label"].shape))

        model = self.create_model()
        for step, (label, image) in enumerate(dataset):
            out = model(image)

            with tf.GradientTape() as tape:

                # loss = tf.reduce_sum(tf.square(out - label)) / image.shape[0]
                # loss = tf.reduce_mean(-tf.reduce_sum(label * (tf.log(softmax) / tf.log(2.)), 1))
                loss = tf.reduce_mean(-tf.reduce_sum(label * (tf.math.log(out) / tf.math.log(2.)), 1))
                print(loss)

            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step == 100:
                break

        # for step, (label, image) in enumerate(dataset):
        #     out = model(image)
        #     print(out)
        #
        #     if step == 0:
        #         break
        #     pass

        # model = keras.models.load_model('my_model.h5')
        #
        # for step, (label, image) in enumerate(dataset):
        #     out = model(image)
        #
        #     print(label)
        #     print(out)
        #     out_see = np.around(out.numpy(), decimals=2)
        #
        #     print(out_see)
        #     if step == 0:
        #         break

        # for step, (label, image) in enumerate(dataset):
        #     with tf.GradientTape() as tape:
        #         out = model(image)
        #         loss = tf.reduce_sum(tf.square(out - label)) / image.shape[0]
        #         print(loss)
        #
        #     grads = tape.gradient(loss, model.trainable_variables)
        #     model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        #
        #     if loss < 0.01:
        #         break

        # print("--end--")
        # model.save('my_model.h5')

    def create_model(self):
        model = tf.keras.models.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation='softmax')
        ])

        # model.compile(optimizer='adam',
        #               loss='sparse_categorical_crossentropy',
        #               metrics=['accuracy'])

        return model


if __name__ == "__main__":
    Main()
