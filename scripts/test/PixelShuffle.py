import numpy as np
from tensorflow.keras.layers import Layer, Input
from numpy.core.fromnumeric import reshape
import tensorflow as tf

class PixelShuffle(Layer):
    def __init__(self, upscaling_factor=2):
        super(PixelShuffle, self).__init__()
        self.scale = upscaling_factor

    def call(self, x):
        [_, height, width, channel] = x.shape

        if _ is None:
            # pass
            y = self.pixelShuffle(tf.squeeze(x, axis=0))
            shape = tf.shape(y)
            op_shape = [-1, shape[0], shape[1], shape[2]]
            y = tf.reshape(y, op_shape)
        else:
            flag = False
            for batch in range(_):
                if not flag:
                    y = self.pixelShuffle(x[batch])
                    shape = tf.shape(y)
                    op_shape = [1, shape[0], shape[1], shape[2]]
                    y = tf.reshape(y, op_shape)
                    flag = True
                else:
                    y_temp = self.pixelShuffle(x[batch])
                    shape = tf.shape(y_temp)
                    op_shape = [1, shape[0], shape[1], shape[2]]
                    y_temp = tf.reshape(y_temp, op_shape)
                    y = tf.concat([y, y_temp], axis=0)
        return y

    def pixelShuffle(self, x):
        [h, w, c] = x.shape
        nh = h*self.scale
        nw = w*self.scale
        nc = c // (self.scale**2)
        y = tf.reshape(x, (nh,nw,nc))
        return y
if __name__ == '__main__':
    pixelLayer = PixelShuffle(2)
    input = Input(shape=(64, 32, 1024))
    output = pixelLayer(input)
    model = tf.keras.Model(inputs=input, outputs=output)
    # model.summary()
    image = np.ones((64,32,1024))
    y = model(image)
    print(y.shape)

