import numpy as np
# import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Dropout, Input

from darknet import Encoder

    
if __name__ == '__main__':
    # decoder = Decoder()
    # self.model = tf.keras.Sequential()
    # model = decoder.createModel()
    # model.summary()

    model = Encoder()
    # model = encoder.createEncoder()
    # model.summary()

    # model2 = encoder.define_skip_model()

    # inp = np.random.randn(2, 5, 64, 1024)
    # out = model(inp)

    model.build((2, 5, 64, 1024))
    model.call(Input(shape=(5, 64, 1024)))
    model.summary()

    # inp = np.array(np.zeros((2, 64, 64, 512)))
    # model2 = model.ResidualBlock(32, inp)
    # model2[1].summary()