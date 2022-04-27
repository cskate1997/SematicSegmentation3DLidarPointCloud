import numpy as np
from tensorflow.keras.layers import Layer, Input
from numpy.core.fromnumeric import reshape
import tensorflow as tf

class PixelShuffle(Layer):
    def __init__(self, upscaling_factor=2):
        super(PixelShuffle, self).__init__()
        self.upscaling_factor = upscaling_factor

    def call(self, x):
        [_, height, width, channel] = x.shape
        n_height = height*self.upscaling_factor
        n_width = width*self.upscaling_factor
        n_channel = channel // (self.upscaling_factor**2)
        y = tf.zeros((_, n_height, n_width, n_channel), dtype=tf.float32)
        for i in range(height):
            for j in range(width):
                y[:, self.upscaling_factor*i:self.upscaling_factor*i+self.upscaling_factor,
                self.upscaling_factor*j:self.upscaling_factor*j+self.upscaling_factor] = tf.reshape(x[:i, j,:], (_, self.upscaling_factor, self.upscaling_factor, n_channel))
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

