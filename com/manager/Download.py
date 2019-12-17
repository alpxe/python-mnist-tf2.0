import os
import urllib.request
import sys

from com.core.base.Singleton import Singleton


class Download(Singleton):
    """Download extends Singleton"""

    __DEFAULT_SOURCE_URL = "http://yann.lecun.com/exdb/mnist/"  # mnist官方地址

    dest_directory = "assets/MNIST_DATA"

    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    def __setattr__(self, key, value):
        """设置成只读"""
        if key == "TEST_IMAGES" or key == "TRAIN_LABELS" or key == "TEST_IMAGES" or key == "TEST_LABELS":
            raise AttributeError('{}.{} is readonly'.format(type(self).__name__, key))
        else:
            self.__dict__[key] = value

    def maybe_download(self, file):
        filepath = os.path.join(self.dest_directory, file)  # 本地文件位置
        url = os.path.join(self.__DEFAULT_SOURCE_URL, file)  # 下载链接

        if not os.path.exists(self.dest_directory):  # 如果文件夹不存在
            os.makedirs(self.dest_directory)  # 创建文件夹
            pass

        if not os.path.exists(filepath):  # 如果文件不存在
            def _progress(count, block_size, total_size):
                sys.stdout.write(
                    '\r>> Downloading %s %.1f%%' % (file, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
                pass

            respath, _ = urllib.request.urlretrieve(url, filepath, _progress)
            statinfo = os.stat(respath)

            print('Successfully downloaded', file, statinfo.st_size, 'bytes.')

    def load(self):
        # 下载mnist数据集
        self.maybe_download(self.TRAIN_IMAGES)
        self.maybe_download(self.TRAIN_LABELS)
        self.maybe_download(self.TEST_IMAGES)
        self.maybe_download(self.TEST_LABELS)
