import numpy as np
# import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, Dropout, Input

from darknet import DarkNet

    
if __name__ == '__main__':
    # decoder = Decoder()
    # self.model = tf.keras.Sequential()
    # model = decoder.createModel()
    # model.summary()

    model = DarkNet()
    # model = encoder.createEncoder()
    # model.summary()

    # model2 = encoder.define_skip_model()

    # inp = np.random.randn(2, 64, 1024, 5)
    # out = model(inp)

    model.build((2, 64, 1024, 5))
    model.summary()
    model.get_layer(name="encoder").summary()

    # inp = np.array(np.zeros((2, 64, 64, 512)))
    # model2 = model.ResidualBlock(32, inp)
    # model2[1].summary()