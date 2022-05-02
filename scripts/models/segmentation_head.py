import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout

from tensorflow.keras.layers import Input, concatenate, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import layers

INPUT_DEPTH = 5
INPUT_HEIGHT = 64
INPUT_WIDTH = 1024

class SegmentationHead(Model):
    def __init__(self):
        super(SegmentationHead, self).__init__()
        self.dropout = Dropout(rate=0.01)
        self.conv = Conv2D(filters=20, kernel_size=3, strides=1, padding='same', data_format='channels_last')

    def call(self, x):
        y = self.dropout(x)
        y = self.conv(y)
        return y

if __name__ == '__main__':
    # tf.enable_eager_execution()

    segmentation_head = SegmentationHead()

    segmentation_head.build(input_shape=(None, 32, 64, 1024))
    segmentation_head.call(Input(shape=(32, 64, 1024)))
    segmentation_head.summary()