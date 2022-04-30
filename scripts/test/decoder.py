import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, LeakyReLU, Conv2DTranspose

from tensorflow.keras.layers import Input, concatenate, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from PixelShuffle import PixelShuffle

INPUT_DEPTH = 5
INPUT_HEIGHT = 64
INPUT_WIDTH = 1024

class BasicBlock(Model):
    def __init__(self, planes, bn_d=0.01, bn_axis=1, data_format='channels_last'):
        super(BasicBlock, self).__init__()
        self.data_format = data_format
        self.conv1 = Conv2D(planes[0], kernel_size = 1, strides = 1, padding = "same", use_bias=False, data_format=data_format)
        self.bn1 = BatchNormalization(axis=bn_axis, momentum=bn_d)
        self.relu1 = LeakyReLU(0.1)

        self.conv2 = Conv2D(planes[1], kernel_size = 3, strides = 1, padding = "same", use_bias=False, data_format=data_format)
        self.bn2 = BatchNormalization(axis=bn_axis, momentum=bn_d)
        self.relu2 = LeakyReLU(0.1)

    def call(self, x):
        res = x
        
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)

        y += res

        return y



class Decoder(Model):
    def __init__(self, pixel_shuffle=False):
        super(Decoder, self).__init__()

        self.strides = [2, 2, 2, 2, 2]
        self.bn_d = 0.01
        self.data_format='channels_last' 
        self.feature_depth=1024
        self.pixel_shuffle_flag = pixel_shuffle

        if self.data_format == 'channels_first':
            self.bn_axis = 1  
        else:
            self.bn_axis = 3

        self.dec5 = self.make_decoder_layer([1024, 512], bn_d=self.bn_d, stride=self.strides[0])
        self.dec4 = self.make_decoder_layer([512, 256], bn_d=self.bn_d, stride=self.strides[1])
        self.dec3 = self.make_decoder_layer([256, 128], bn_d=self.bn_d, stride=self.strides[2])
        self.dec2 = self.make_decoder_layer([128, 64], bn_d=self.bn_d, stride=self.strides[3])
        self.dec1 = self.make_decoder_layer([64, 32], bn_d=self.bn_d, stride=self.strides[4])

        self.dropout = Dropout(rate=0.01)

    def call(self, x):
        skips = self.skips
        os = self.os

        y = self.dec5[0](x)

        for i in range(3):
            y = self.dec5[1+i](y)
        y, skips, os = self.add_skip(x, y, skips, os)

        x = y
        for i in range(4):
            y = self.dec4[i](y)
        y, skips, os = self.add_skip(x, y, skips, os)
        
        x = y
        for i in range(4):
            y = self.dec3[i](y)
        y, skips, os = self.add_skip(x, y, skips, os)
        
        x = y
        for i in range(4):
            y = self.dec2[i](y)
        y, skips, os = self.add_skip(x, y, skips, os)
        
        x = y
        for i in range(4):
            y = self.dec1[i](y)
        y, skips, os = self.add_skip(x, y, skips, os)

        y = self.dropout(y)

        return y
    
    def add_skip(self, x, y, skips, os):
        if y.shape[2] > x.shape[2]:
            os //= 2
            y = y + tf.stop_gradient(skips[os])
        return y, skips, os

    def set_skips(self, skips, os):
        self.skips = skips
        self.os = os

    def make_decoder_layer(self, planes, bn_d=0.01, stride= 2):
        layers = []

        if stride == 2:
            if self.pixel_shuffle_flag:
                layers.append(PixelShuffle())
                layers.append(Conv2D(planes[1], kernel_size = 3, padding='same', data_format=self.data_format))
            else:
                layers.append(Conv2DTranspose(planes[1], kernel_size = [1, 4], strides = [1, 2], padding='same', data_format=self.data_format))
        else:
            layers.append(Conv2D(planes[1], kernel_size = 3, padding='same', data_format=self.data_format))
        
        layers.append(BatchNormalization(axis=self.bn_axis, momentum=bn_d))
        layers.append(LeakyReLU(0.1))

        layers.append(BasicBlock(planes, bn_d, self.bn_axis))

        return layers


if __name__ == '__main__':
    # tf.enable_eager_execution()

    decoder = Decoder(pixel_shuffle=False)
    decoder.skips = {}
    decoder.os = 32

    decoder.build(input_shape=(None, 64, 32, 1024))
    decoder.call(Input(shape=(64, 32, 1024)))
    decoder.summary()