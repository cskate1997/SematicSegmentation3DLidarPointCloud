import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, LeakyReLU, Conv2DTranspose

from tensorflow.keras.layers import Input, concatenate, UpSampling2D
from tensorflow.keras.models import Model

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

class Encoder(Model):
    def __init__(self, pixel_shuffle=False):
        super(Encoder, self).__init__()

        self.strides = [2, 2, 2, 2, 2]
        self.blocks = [1, 2, 8, 8, 4]
        self.bn_d = 0.01
        self.data_format='channels_last' 
        self.feature_depth=1024
        self.pixel_shuffle_flag = pixel_shuffle

        if self.data_format == 'channels_first':
            self.bn_axis = 1  
        else:
            self.bn_axis = 3


        self.conv0 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', data_format=self.data_format, use_bias=False, input_shape=(64, 1024, 5))
        self.bn0 = BatchNormalization(axis = self.bn_axis, momentum=self.bn_d)
        self.lrelu0 = LeakyReLU(alpha=0.1)

        self.enc1 = self.make_encoder_layer([32, 64], blocks=self.blocks[0], stride=self.strides[0], bn_d=self.bn_d)
        self.drop1 = Dropout(rate=0.01)
        self.enc2 = self.make_encoder_layer([64, 128], blocks=self.blocks[1], stride=self.strides[1], bn_d=self.bn_d)
        self.drop2 = Dropout(rate=0.01)
        self.enc3 = self.make_encoder_layer([128, 256], blocks=self.blocks[2], stride=self.strides[2], bn_d=self.bn_d)
        self.drop3 = Dropout(rate=0.01)
        self.enc4 = self.make_encoder_layer([256, 512], blocks=self.blocks[3], stride=self.strides[3], bn_d=self.bn_d)
        self.drop4 = Dropout(rate=0.01)
        self.enc5 = self.make_encoder_layer([512, 1024], blocks=self.blocks[4], stride=self.strides[4], bn_d=self.bn_d)
        self.drop5 = Dropout(rate=0.01)

    def append_skip(self, x, y, skips, os):
        if y.shape[1] < x.shape[1] or y.shape[2] < x.shape[2]:
            skips[os] = tf.stop_gradient(x)
            os *= 2
        return skips, os

    def call(self, x):
        skips = {}
        os = 1
        y = self.conv0(x)
        y = self.bn0(y)
        y = self.lrelu0(y)

        x = y
        for i in range(4):
            y = self.enc1[i](y)
        skips, os = self.append_skip(x, y, skips, os)
        y = self.drop1(y)

        x = y
        for i in range(5):
            y = self.enc2[i](y)
        skips, os = self.append_skip(x, y, skips, os)
        y = self.drop2(y)

        x = y
        for i in range(11):
            y = self.enc3[i](y)
        skips, os = self.append_skip(x, y, skips, os)
        y = self.drop3(y)

        x = y
        for i in range(11):
            y = self.enc4[i](y)
        skips, os = self.append_skip(x, y, skips, os)
        y = self.drop4(y)

        x = y
        for i in range(7):
            y = self.enc5[i](y)
        skips, os = self.append_skip(x, y, skips, os)
        y = self.drop5(y)

        self.skips = skips
        self.os = os
        return y

    def get_skips(self):
        return self.skips, self.os

    def make_encoder_layer(self, planes, blocks, stride, bn_d=0.1):
        layers = []

        if self.pixel_shuffle_flag:
            layers.append(Conv2D(filters=planes[1], kernel_size=3, dilation_rate=1, strides=[stride, stride], padding='same', data_format=self.data_format, use_bias=False))
        else:
            layers.append(Conv2D(filters=planes[1], kernel_size=3, dilation_rate=1, strides=[1, stride], padding='same', data_format=self.data_format, use_bias=False))
        layers.append(BatchNormalization(axis = self.bn_axis, momentum=self.bn_d))
        layers.append(LeakyReLU(0.1))

        for i in range(0, blocks):
            layers.append(BasicBlock(planes, bn_d, self.bn_axis))

        return layers

if __name__ == '__main__':
    # tf.enable_eager_execution()

    encoder = Encoder(pixel_shuffle=False)

    encoder.build(input_shape=(None, 64, 1024, 5))
    encoder.call(Input(shape=(64, 1024, 5)))
    encoder.summary()