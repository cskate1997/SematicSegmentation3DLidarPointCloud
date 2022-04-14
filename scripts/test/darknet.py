import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Dense, GlobalAveragePooling2D, BatchNormalization, LeakyReLU, Dropout

from keras.layers import Input, concatenate, UpSampling2D
from keras.models import Model
from keras import layers

INPUT_DEPTH = 5
INPUT_HEIGHT = 64
INPUT_WIDTH = 1024

class DarkNet(Model):
    def __init__(self):
        super(DarkNet, self).__init__()
        print("Using DarkNet Backbone")

        # self.conv_block = tf.keras.Sequential()
        # self.residual_block = tf.keras.Sequential()
        
        # self.residual_reps = [1, 2, 8, 8, 4]
        # self.conv_filters = [64, 128, 256, 512, 1024]
        # self.res_filters = [32, 64, 128, 256, 512]
        # self.count = 0
        # self.bn_momentum = 0.01
        # self.conv2 = Conv2D(filters=64, kernel_size=3, dilation_rate=1, strides=(1, 2), padding='same', data_format='channels_last', use_bias=False)
        # self.createConv(64)

        self.encode = Encoder()

    def call(self, input):
        y = self.encode(input)
        # y = 0
        return y


# class Encoder(layers.Layer):

class Encoder(Model):
    def __init__(self, inp=(2, 64, 1024, 5)):
        super(Encoder, self).__init__()
        self.bn_momentum = 0.01
        self.residual_reps = [1, 2, 8, 8, 4]
        self.conv_filters = [64, 128, 256, 512, 1024]
        self.res_filters = [32, 64, 128, 256, 512]
        # self.count = 0

        self.conv1 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', 
            input_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_DEPTH), data_format='channels_last', use_bias=False)
        self.bn1 = BatchNormalization(axis = 3, momentum=self.bn_momentum)
        self.lrelu1 = LeakyReLU(alpha=0.1)

        self.createConv(64)
        
        # self.conv1 = Conv2D(filters=64, kernel_size=3, dilation_rate=1, strides=(1, 2), padding='same', data_format='channels_last', use_bias=False, input_shape=inp)
        # self.bn2 = BatchNormalization(axis = 3, momentum=0.01)
        # self.lrelu2 = LeakyReLU(alpha=0.1)

    def call(self, input):
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.lrelu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.lrelu2(out)
        return out

    def createConv(self, filters):
    # Create Convolution Layer used before each Residual (Basic) Block
        self.conv2 = Conv2D(filters=filters, kernel_size=3, dilation_rate=1, strides=(1, 2), padding='same', data_format='channels_last', use_bias=False)
        self.bn2 = BatchNormalization(axis = 3, momentum=self.bn_momentum)
        self.lrelu2 = LeakyReLU(alpha=0.1)

    def ResidualBlock(self, filters, x):
    # Create Residual Block which performs Convolution with 1x1 Kernel, followed by
    # Convolution with 3x3 Kernel
    # Input before Convolution is added to Output after Convolution
        input = Input(x.shape)
        residual = input

        out1 = Conv2D(filters=filters, kernel_size=1, strides=1, padding='valid', data_format='channels_last', use_bias=False)(input)
        out2 = BatchNormalization(axis = 3, momentum=self.bn_momentum)(out1)
        out3 = LeakyReLU(alpha=0.1)(out2)
        out4 = Conv2D(filters=filters * 2, kernel_size=3, strides=1, padding='same', data_format='channels_last', use_bias=False)(out3)
        out5 = BatchNormalization(axis = 3, momentum=self.bn_momentum)(out4)
        out6 = LeakyReLU(alpha=0.1)(out5)
        self.count += 6

        print("=================")
        print(out6.shape, residual.shape)
        out6 += residual

        model = Model(inputs=input, outputs=out6)
        return out6, model
    
    def make_encoder_layer():
    # Convolution Layer + Residual
        pass

    def createEncoder(self):
        self.count = 0

        # Input Layer
        self.model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', 
            input_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_DEPTH), data_format='channels_last', use_bias=False))
        self.model.add(BatchNormalization(axis = 3, momentum=self.bn_momentum))
        self.model.add(LeakyReLU(alpha=0.1))
        self.count = 3

        for i in range(5):
            self.createConv(filters=self.conv_filters[i])
            self.createResidual(filters=self.res_filters[i], n_reps=self.residual_reps[i])
            self.model.add(Dropout(0.01))
            self.count+=1

        self.model.compile()
        self.model.build()
        self.model.summary()
        print("Count: ", self.count)
        return self.model

    def createModel(self):
    # Encoder + Decoder
        pass

    def define_skip_model(self):  
        input_net = Input((32,32,3))
        
        ## Encoder starts
        conv1 = Conv2D(32, 3, strides=(2,2), activation = 'relu', padding = 'same')(input_net)
        conv2 = Conv2D(64, 3, strides=(2,2), activation = 'relu', padding = 'same')(conv1)
        conv3 = Conv2D(128, 3, strides=(2,2), activation = 'relu', padding = 'same')(conv2)
        
        conv4 = Conv2D(128, 3, strides=(2,2), activation = 'relu', padding = 'same')(conv3)
        
        ## And now the decoder
        up1 = Conv2D(128, 3, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv4))
        merge1 = concatenate([conv3,up1], axis = 3)
        up2 = Conv2D(64, 3, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(merge1))
        merge2 = concatenate([conv2,up2], axis = 3)
        up3 = Conv2D(32, 3, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(merge2))
        merge3 = concatenate([conv1,up3], axis = 3)
        
        up4 = Conv2D(32, 3, padding = 'same')(UpSampling2D(size = (2,2))(merge3))
        
        output_net = Conv2D(3, 3, padding = 'same')(up4)
        
        model = Model(inputs = input_net, outputs = output_net)
        
        return model

# if __name__ == '__main__':
#     encoder = DarkNet()
#     # model = encoder.createEncoder()
#     # model.summary()

#     # model2 = encoder.define_skip_model()

#     inp = np.random.randn(2, 64, 512, 64)
#     # inp = np.array(np.zeros((2, 64, 64, 512)))
#     model2 = encoder.ResidualBlock(32, inp)
#     model2[1].summary()