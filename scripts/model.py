import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

def createModel():
    model = tf.keras.Sequential()
    model.add(Conv2D(filters=))
    return model

if __name__ == '__main__':
    model = createModel()
    model.summary() 