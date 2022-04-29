from shutil import ExecError
import numpy as np
from tensorflow.keras.layers import Layer, Input
from numpy.core.fromnumeric import reshape
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import colors

class PixelShuffle(Layer):
    def __init__(self, upscaling_factor=2):
        super(PixelShuffle, self).__init__()
        self.scale = upscaling_factor

    def call(self, x):
        [_, height, width, channel] = x.shape
        y = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], self.scale, self.scale, channel//(self.scale**2)))
        y = tf.transpose(y, (0, 1, 3, 2, 4, 5))
        y = tf.reshape(y, (tf.shape(x)[0], height*self.scale, width*self.scale, channel//(self.scale**2)))
        return y

if __name__ == '__main__':
    pixelLayer = PixelShuffle(2)
    input = Input(shape=(64, 32, 8))
    output = pixelLayer(input)
    model = tf.keras.Model(inputs=input, outputs=output)
    model.summary()
    image = np.ones((1, 64,32,8))
    for i in range(0,8):
        image[0,:,:,i] = (i+1)
    y = model(image)
    print(y.shape)
    y = np.reshape(y, (128,64,-1))
    colormap = colors.ListedColormap(["red","#39FF14","blue","yellow"])
    plt.figure()
    plt.imshow(y[:,:,0], cmap=colormap)
    plt.show()

