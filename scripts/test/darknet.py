import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, GlobalAveragePooling2D, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.models import Model

# from tensorflow.keras.layers import Input, concatenate, UpSampling2D
# from tensorflow.keras import layers

INPUT_DEPTH = 5
INPUT_HEIGHT = 64
INPUT_WIDTH = 1024

# class DarkNet(Model):
#     def __init__(self):
#         super(DarkNet, self).__init__()
#         print("Using DarkNet Backbone")

#         # self.conv_block = tf.keras.Sequential()
#         # self.residual_block = tf.keras.Sequential()
        
#         # self.residual_reps = [1, 2, 8, 8, 4]
#         # self.conv_filters = [64, 128, 256, 512, 1024]
#         # self.res_filters = [32, 64, 128, 256, 512]
#         # self.count = 0
#         # self.bn_momentum = 0.01
#         # self.conv2 = Conv2D(filters=64, kernel_size=3, dilation_rate=1, strides=(1, 2), padding='same', data_format=self.data_format, use_bias=False)
#         # self.createConv(64)

#         self.encode = Encoder()

#     def call(self, input):
#         y = self.encode(input)
#         # y = 0
#         return y

class Encoder(Model):
    def __init__(self, inp=(5, 64, 1024)):
        super(Encoder, self).__init__()
        self.bn_momentum = 0.01
        self.residual_reps = [1, 2, 8, 8, 4]
        self.conv_filters = [64, 128, 256, 512, 1024]
        self.res_filters = [32, 64, 128, 256, 512]
        self.dec_conv_filters = [1024, 512, 156, 128, 64]
        self.dec_res_filters = [512, 256, 128, 64, 32]
        self.blocks = len(self.conv_filters)
        self.skips = []        
        self.data_format = 'channels_first'

        if self.data_format == 'channels_first':
            self.bn_axis = 1
        else:
            self.bn_axis = 3

        ################## ENCODER ##################
        self.conv0 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', 
                input_shape=(INPUT_DEPTH, INPUT_HEIGHT, INPUT_WIDTH), data_format=self.data_format, use_bias=False)
        self.bn0 = BatchNormalization(axis = self.bn_axis, momentum=self.bn_momentum)
        self.lrelu0 = LeakyReLU(alpha=0.1)

        # self.ds_layers = []
        # self.res_layers = []
        # self.drop_layers = []

        # for i in range(0, 1):
        #     [self.conv1, self.bn1, self.lrelu1] = self.create_Conv(self.conv_filters[i], i)
            # for n_reps in range(self.residual_reps[i]):
                # self.create_ResidualBlock(self.res_filters[i], i)
            # self.create_DropOut()
        
        ## BLOCK 1 ##
        [self.bk1_conv1, self.bk1_bn1, self.bk1_lrelu1] = self.create_Conv(self.conv_filters[0], 0)
        [self.bk1_res1_conv1, self.bk1_res1_bn1, self.bk1_res1_lr1, self.bk1_res1_conv2, self.bk1_res1_bn2, self.bk1_res1_lr2] = self.create_ResidualBlock(self.res_filters[0], 1)
        self.enc_drop1 = self.create_DropOut()

        ## BLOCK 2 ##
        [self.bk2_conv1, self.bk2_bn1, self.bk2_lrelu1] = self.create_Conv(self.conv_filters[1], 1)
        [self.bk2_res1_conv1, self.bk2_res1_bn1, self.bk2_res1_lr1, self.bk2_res1_conv2, self.bk2_res1_bn2, self.bk2_res1_lr2] = self.create_ResidualBlock(self.res_filters[1], 1)
        [self.bk2_res2_conv1, self.bk2_res2_bn1, self.bk2_res2_lr1, self.bk2_res2_conv2, self.bk2_res2_bn2, self.bk2_res2_lr2] = self.create_ResidualBlock(self.res_filters[1], 1)
        self.enc_drop2 = self.create_DropOut()

        ## BLOCK 3 ##
        [self.bk3_conv1, self.bk3_bn1, self.bk3_lrelu1] = self.create_Conv(self.conv_filters[2], 2)
        [self.bk3_res1_conv1, self.bk3_res1_bn1, self.bk3_res1_lr1, self.bk3_res1_conv2, self.bk3_res1_bn2, self.bk3_res1_lr2] = self.create_ResidualBlock(self.res_filters[2], 1)
        [self.bk3_res2_conv1, self.bk3_res2_bn1, self.bk3_res2_lr1, self.bk3_res2_conv2, self.bk3_res2_bn2, self.bk3_res2_lr2] = self.create_ResidualBlock(self.res_filters[2], 1)
        [self.bk3_res3_conv1, self.bk3_res3_bn1, self.bk3_res3_lr1, self.bk3_res3_conv2, self.bk3_res3_bn2, self.bk3_res3_lr2] = self.create_ResidualBlock(self.res_filters[2], 1)
        [self.bk3_res4_conv1, self.bk3_res4_bn1, self.bk3_res4_lr1, self.bk3_res4_conv2, self.bk3_res4_bn2, self.bk3_res4_lr2] = self.create_ResidualBlock(self.res_filters[2], 1)
        [self.bk3_res5_conv1, self.bk3_res5_bn1, self.bk3_res5_lr1, self.bk3_res5_conv2, self.bk3_res5_bn2, self.bk3_res5_lr2] = self.create_ResidualBlock(self.res_filters[2], 1)
        [self.bk3_res6_conv1, self.bk3_res6_bn1, self.bk3_res6_lr1, self.bk3_res6_conv2, self.bk3_res6_bn2, self.bk3_res6_lr2] = self.create_ResidualBlock(self.res_filters[2], 1)
        [self.bk3_res7_conv1, self.bk3_res7_bn1, self.bk3_res7_lr1, self.bk3_res7_conv2, self.bk3_res7_bn2, self.bk3_res7_lr2] = self.create_ResidualBlock(self.res_filters[2], 1)
        [self.bk3_res8_conv1, self.bk3_res8_bn1, self.bk3_res8_lr1, self.bk3_res8_conv2, self.bk3_res8_bn2, self.bk3_res8_lr2] = self.create_ResidualBlock(self.res_filters[2], 1)
        self.enc_drop3 = self.create_DropOut()

        ## BLOCK 4 ##
        [self.bk4_conv1, self.bk4_bn1, self.bk4_lrelu1] = self.create_Conv(self.conv_filters[3], 3)
        [self.bk4_res1_conv1, self.bk4_res1_bn1, self.bk4_res1_lr1, self.bk4_res1_conv2, self.bk4_res1_bn2, self.bk4_res1_lr2] = self.create_ResidualBlock(self.res_filters[3], 1)
        [self.bk4_res2_conv1, self.bk4_res2_bn1, self.bk4_res2_lr1, self.bk4_res2_conv2, self.bk4_res2_bn2, self.bk4_res2_lr2] = self.create_ResidualBlock(self.res_filters[3], 1)
        [self.bk4_res3_conv1, self.bk4_res3_bn1, self.bk4_res3_lr1, self.bk4_res3_conv2, self.bk4_res3_bn2, self.bk4_res3_lr2] = self.create_ResidualBlock(self.res_filters[3], 1)
        [self.bk4_res4_conv1, self.bk4_res4_bn1, self.bk4_res4_lr1, self.bk4_res4_conv2, self.bk4_res4_bn2, self.bk4_res4_lr2] = self.create_ResidualBlock(self.res_filters[3], 1)
        [self.bk4_res5_conv1, self.bk4_res5_bn1, self.bk4_res5_lr1, self.bk4_res5_conv2, self.bk4_res5_bn2, self.bk4_res5_lr2] = self.create_ResidualBlock(self.res_filters[3], 1)
        [self.bk4_res6_conv1, self.bk4_res6_bn1, self.bk4_res6_lr1, self.bk4_res6_conv2, self.bk4_res6_bn2, self.bk4_res6_lr2] = self.create_ResidualBlock(self.res_filters[3], 1)
        [self.bk4_res7_conv1, self.bk4_res7_bn1, self.bk4_res7_lr1, self.bk4_res7_conv2, self.bk4_res7_bn2, self.bk4_res7_lr2] = self.create_ResidualBlock(self.res_filters[3], 1)
        [self.bk4_res8_conv1, self.bk4_res8_bn1, self.bk4_res8_lr1, self.bk4_res8_conv2, self.bk4_res8_bn2, self.bk4_res8_lr2] = self.create_ResidualBlock(self.res_filters[3], 1)
        self.enc_drop4 = self.create_DropOut()

        ## BLOCK 5 ##
        [self.bk5_conv1, self.bk5_bn1, self.bk5_lrelu1] = self.create_Conv(self.conv_filters[4], 4)
        [self.bk5_res1_conv1, self.bk5_res1_bn1, self.bk5_res1_lr1, self.bk5_res1_conv2, self.bk5_res1_bn2, self.bk5_res1_lr2] = self.create_ResidualBlock(self.res_filters[4], 1)
        [self.bk5_res2_conv1, self.bk5_res2_bn1, self.bk5_res2_lr1, self.bk5_res2_conv2, self.bk5_res2_bn2, self.bk5_res2_lr2] = self.create_ResidualBlock(self.res_filters[4], 1)
        [self.bk5_res3_conv1, self.bk5_res3_bn1, self.bk5_res3_lr1, self.bk5_res3_conv2, self.bk5_res3_bn2, self.bk5_res3_lr2] = self.create_ResidualBlock(self.res_filters[4], 1)
        [self.bk5_res4_conv1, self.bk5_res4_bn1, self.bk5_res4_lr1, self.bk5_res4_conv2, self.bk5_res4_bn2, self.bk5_res4_lr2] = self.create_ResidualBlock(self.res_filters[4], 1)
        self.enc_drop5 = self.create_DropOut()


        ################## DECODER ##################
        ## BLOCK 1 ##
        [self.dec1_conv1, self.dec1_bn1, self.dec1_lrelu1] = self.create_Decoder_Conv(self.dec_res_filters[0], 0)
        [self.dec1_res1_conv1, self.dec1_res1_bn1, self.dec1_res1_lr1, self.dec1_res1_conv2, self.dec1_res1_bn2, self.dec1_res1_lr2] = self.create_ResidualBlock(self.dec_conv_filters[0], 0, factor=0.5)
        # self.dec_drop1 = self.create_DropOut()

        ## BLOCK 2 ##
        [self.dec2_conv1, self.dec2_bn1, self.dec2_lrelu1] = self.create_Decoder_Conv(self.dec_res_filters[1], 1)
        [self.dec2_res1_conv1, self.dec2_res1_bn1, self.dec2_res1_lr1, self.dec2_res1_conv2, self.dec2_res1_bn2, self.dec2_res1_lr2] = self.create_ResidualBlock(self.dec_conv_filters[1], 1, factor=0.5)
        # self.dec_drop2 = self.create_DropOut()

        ## BLOCK 3 ##
        [self.dec3_conv1, self.dec3_bn1, self.dec3_lrelu1] = self.create_Decoder_Conv(self.dec_res_filters[2], 2)
        [self.dec3_res1_conv1, self.dec3_res1_bn1, self.dec3_res1_lr1, self.dec3_res1_conv2, self.dec3_res1_bn2, self.dec3_res1_lr2] = self.create_ResidualBlock(self.dec_conv_filters[2], 2, factor=0.5)
        # self.dec_drop3 = self.create_DropOut()

        ## BLOCK 4 ##
        [self.dec4_conv1, self.dec4_bn1, self.dec4_lrelu1] = self.create_Decoder_Conv(self.dec_res_filters[3], 3)
        [self.dec4_res1_conv1, self.dec4_res1_bn1, self.dec4_res1_lr1, self.dec4_res1_conv2, self.dec4_res1_bn2, self.dec4_res1_lr2] = self.create_ResidualBlock(self.dec_conv_filters[3], 3, factor=0.5)
        # self.dec_drop4 = self.create_DropOut()

        ## BLOCK 5 ##
        [self.dec5_conv1, self.dec5_bn1, self.dec5_lrelu1] = self.create_Decoder_Conv(self.dec_res_filters[4], 4)
        [self.dec5_res1_conv1, self.dec5_res1_bn1, self.dec5_res1_lr1, self.dec5_res1_conv2, self.dec5_res1_bn2, self.dec5_res1_lr2] = self.create_ResidualBlock(self.dec_conv_filters[4], 4, factor=0.5)
        # self.dec_drop5 = self.create_DropOut()

        

    def call(self, input):
        out = self.conv0(input)
        out = self.bn0(out)
        out = self.lrelu0(out)

        ## BLOCK 1 ##
        self.skips.append(out)

        out = self.bk1_conv1(out)
        out = self.bk1_bn1(out)
        out = self.bk1_lrelu1(out)

        residual = out
        out = self.bk1_res1_conv1(out)
        out = self.bk1_res1_bn1(out)
        out = self.bk1_res1_lr1(out)
        out = self.bk1_res1_conv2(out)
        out = self.bk1_res1_bn2(out)
        out = self.bk1_res1_lr2(out)
        out += residual

        out = self.enc_drop1(out)

        ## BLOCK 2 ##
        self.skips.append(out)

        out = self.bk2_conv1(out)
        out = self.bk2_bn1(out)
        out = self.bk2_lrelu1(out)

        residual = out
        out = self.bk2_res1_conv1(out)
        out = self.bk2_res1_bn1(out)
        out = self.bk2_res1_lr1(out)
        out = self.bk2_res1_conv2(out)
        out = self.bk2_res1_bn2(out)
        out = self.bk2_res1_lr2(out)
        out += residual

        residual = out
        out = self.bk2_res2_conv1(out)
        out = self.bk2_res2_bn1(out)
        out = self.bk2_res2_lr1(out)
        out = self.bk2_res2_conv2(out)
        out = self.bk2_res2_bn2(out)
        out = self.bk2_res2_lr2(out)
        out += residual

        out = self.enc_drop2(out)

        ## BLOCK 3 ##
        self.skips.append(out)

        out = self.bk3_conv1(out)
        out = self.bk3_bn1(out)
        out = self.bk3_lrelu1(out)

        residual = out
        out = self.bk3_res1_conv1(out)
        out = self.bk3_res1_bn1(out)
        out = self.bk3_res1_lr1(out)
        out = self.bk3_res1_conv2(out)
        out = self.bk3_res1_bn2(out)
        out = self.bk3_res1_lr2(out)
        out += residual

        residual = out
        out = self.bk3_res2_conv1(out)
        out = self.bk3_res2_bn1(out)
        out = self.bk3_res2_lr1(out)
        out = self.bk3_res2_conv2(out)
        out = self.bk3_res2_bn2(out)
        out = self.bk3_res2_lr2(out)
        out += residual

        residual = out
        out = self.bk3_res3_conv1(out)
        out = self.bk3_res3_bn1(out)
        out = self.bk3_res3_lr1(out)
        out = self.bk3_res3_conv2(out)
        out = self.bk3_res3_bn2(out)
        out = self.bk3_res3_lr2(out)
        out += residual

        residual = out
        out = self.bk3_res4_conv1(out)
        out = self.bk3_res4_bn1(out)
        out = self.bk3_res4_lr1(out)
        out = self.bk3_res4_conv2(out)
        out = self.bk3_res4_bn2(out)
        out = self.bk3_res4_lr2(out)
        out += residual

        residual = out
        out = self.bk3_res5_conv1(out)
        out = self.bk3_res5_bn1(out)
        out = self.bk3_res5_lr1(out)
        out = self.bk3_res5_conv2(out)
        out = self.bk3_res5_bn2(out)
        out = self.bk3_res5_lr2(out)
        out += residual

        residual = out
        out = self.bk3_res6_conv1(out)
        out = self.bk3_res6_bn1(out)
        out = self.bk3_res6_lr1(out)
        out = self.bk3_res6_conv2(out)
        out = self.bk3_res6_bn2(out)
        out = self.bk3_res6_lr2(out)
        out += residual

        residual = out
        out = self.bk3_res7_conv1(out)
        out = self.bk3_res7_bn1(out)
        out = self.bk3_res7_lr1(out)
        out = self.bk3_res7_conv2(out)
        out = self.bk3_res7_bn2(out)
        out = self.bk3_res7_lr2(out)
        out += residual

        residual = out
        out = self.bk3_res8_conv1(out)
        out = self.bk3_res8_bn1(out)
        out = self.bk3_res8_lr1(out)
        out = self.bk3_res8_conv2(out)
        out = self.bk3_res8_bn2(out)
        out = self.bk3_res8_lr2(out)
        out += residual

        out = self.enc_drop3(out)

        ## BLOCK 4 ##
        self.skips.append(out)

        out = self.bk4_conv1(out)
        out = self.bk4_bn1(out)
        out = self.bk4_lrelu1(out)

        residual = out
        out = self.bk4_res1_conv1(out)
        out = self.bk4_res1_bn1(out)
        out = self.bk4_res1_lr1(out)
        out = self.bk4_res1_conv2(out)
        out = self.bk4_res1_bn2(out)
        out = self.bk4_res1_lr2(out)
        out += residual

        residual = out
        out = self.bk4_res2_conv1(out)
        out = self.bk4_res2_bn1(out)
        out = self.bk4_res2_lr1(out)
        out = self.bk4_res2_conv2(out)
        out = self.bk4_res2_bn2(out)
        out = self.bk4_res2_lr2(out)
        out += residual

        residual = out
        out = self.bk4_res3_conv1(out)
        out = self.bk4_res3_bn1(out)
        out = self.bk4_res3_lr1(out)
        out = self.bk4_res3_conv2(out)
        out = self.bk4_res3_bn2(out)
        out = self.bk4_res3_lr2(out)
        out += residual

        residual = out
        out = self.bk4_res4_conv1(out)
        out = self.bk4_res4_bn1(out)
        out = self.bk4_res4_lr1(out)
        out = self.bk4_res4_conv2(out)
        out = self.bk4_res4_bn2(out)
        out = self.bk4_res4_lr2(out)
        out += residual

        residual = out
        out = self.bk4_res5_conv1(out)
        out = self.bk4_res5_bn1(out)
        out = self.bk4_res5_lr1(out)
        out = self.bk4_res5_conv2(out)
        out = self.bk4_res5_bn2(out)
        out = self.bk4_res5_lr2(out)
        out += residual

        residual = out
        out = self.bk4_res6_conv1(out)
        out = self.bk4_res6_bn1(out)
        out = self.bk4_res6_lr1(out)
        out = self.bk4_res6_conv2(out)
        out = self.bk4_res6_bn2(out)
        out = self.bk4_res6_lr2(out)
        out += residual

        residual = out
        out = self.bk4_res7_conv1(out)
        out = self.bk4_res7_bn1(out)
        out = self.bk4_res7_lr1(out)
        out = self.bk4_res7_conv2(out)
        out = self.bk4_res7_bn2(out)
        out = self.bk4_res7_lr2(out)
        out += residual

        residual = out
        out = self.bk4_res8_conv1(out)
        out = self.bk4_res8_bn1(out)
        out = self.bk4_res8_lr1(out)
        out = self.bk4_res8_conv2(out)
        out = self.bk4_res8_bn2(out)
        out = self.bk4_res8_lr2(out)
        out += residual

        out = self.enc_drop4(out)

        ## BLOCK 5 ##
        self.skips.append(out)

        out = self.bk5_conv1(out)
        out = self.bk5_bn1(out)
        out = self.bk5_lrelu1(out)

        residual = out
        out = self.bk5_res1_conv1(out)
        out = self.bk5_res1_bn1(out)
        out = self.bk5_res1_lr1(out)
        out = self.bk5_res1_conv2(out)
        out = self.bk5_res1_bn2(out)
        out = self.bk5_res1_lr2(out)
        out += residual

        residual = out
        out = self.bk5_res2_conv1(out)
        out = self.bk5_res2_bn1(out)
        out = self.bk5_res2_lr1(out)
        out = self.bk5_res2_conv2(out)
        out = self.bk5_res2_bn2(out)
        out = self.bk5_res2_lr2(out)
        out += residual

        residual = out
        out = self.bk5_res3_conv1(out)
        out = self.bk5_res3_bn1(out)
        out = self.bk5_res3_lr1(out)
        out = self.bk5_res3_conv2(out)
        out = self.bk5_res3_bn2(out)
        out = self.bk5_res3_lr2(out)
        out += residual

        residual = out
        out = self.bk5_res4_conv1(out)
        out = self.bk5_res4_bn1(out)
        out = self.bk5_res4_lr1(out)
        out = self.bk5_res4_conv2(out)
        out = self.bk5_res4_bn2(out)
        out = self.bk5_res4_lr2(out)
        out += residual

        out = self.enc_drop5(out)

        print("ENCODER OUTPUT: ", out.shape)

        # for i in range(self.blocks):
        #     self.skips.append(out)
        #     ds = self.ds_layers[i]
        #     for ds_layer in ds:
        #         out = ds_layer(out)
            
        #     res = self.res_layers[i]
        #     for n_reps in range(self.residual_reps[i]):
        #         residual = out
        #         for res_layer in res:
        #             out = res_layer(out)
        #         out += residual
        #     drop = self.drop_layers[i]
        #     out = drop(out)

        # print("SKIPS: ", self.skips)

        ################## DECODER ##################

        ## BLOCK 1 ##
        out = self.dec1_conv1(out)
        out = self.dec1_bn1(out)
        out = self.dec1_lrelu1(out)
        
        residual = out
        out = self.dec1_res1_conv1(out)
        out = self.dec1_res1_bn1(out)
        out = self.dec1_res1_lr1(out)
        out = self.dec1_res1_conv2(out)
        out = self.dec1_res1_bn2(out)
        out = self.dec1_res1_lr2(out)
        out += residual

        # out = self.dec_drop1(out)

        ## BLOCK 2 ##
        out = self.dec2_conv1(out)
        out = self.dec2_bn1(out)
        out = self.dec2_lrelu1(out)
        
        residual = out
        out = self.dec2_res1_conv1(out)
        out = self.dec2_res1_bn1(out)
        out = self.dec2_res1_lr1(out)
        out = self.dec2_res1_conv2(out)
        out = self.dec2_res1_bn2(out)
        out = self.dec2_res1_lr2(out)
        out += residual

        # out = self.dec_drop2(out)

        ## BLOCK 3 ##
        out = self.dec3_conv1(out)
        out = self.dec3_bn1(out)
        out = self.dec3_lrelu1(out)
        
        residual = out
        out = self.dec3_res1_conv1(out)
        out = self.dec3_res1_bn1(out)
        out = self.dec3_res1_lr1(out)
        out = self.dec3_res1_conv2(out)
        out = self.dec3_res1_bn2(out)
        out = self.dec3_res1_lr2(out)
        print("SHAPES: ", out.shape, residual.shape)
        out += residual

        # out = self.dec_drop3(out)

        ## BLOCK 4 ##
        out = self.dec4_conv1(out)
        out = self.dec4_bn1(out)
        out = self.dec4_lrelu1(out)
        
        residual = out
        out = self.dec4_res1_conv1(out)
        out = self.dec4_res1_bn1(out)
        out = self.dec4_res1_lr1(out)
        out = self.dec4_res1_conv2(out)
        out = self.dec4_res1_bn2(out)
        out = self.dec4_res1_lr2(out)
        out += residual

        # out = self.dec_drop4(out)

        ## BLOCK 5 ##
        out = self.dec5_conv1(out)
        out = self.dec5_bn1(out)
        out = self.dec5_lrelu1(out)
        
        residual = out
        out = self.dec5_res1_conv1(out)
        out = self.dec5_res1_bn1(out)
        out = self.dec5_res1_lr1(out)
        out = self.dec5_res1_conv2(out)
        out = self.dec5_res1_bn2(out)
        out = self.dec5_res1_lr2(out)
        out += residual

        # out = self.dec_drop5(out)

        return out
    
    def create_Conv(self, filters, block_num):
    # Create Convolution Layer used before each Residual (Basic) Block
        conv = Conv2D(filters=filters, kernel_size=3, dilation_rate=1, strides=(1, 2), padding='same', data_format=self.data_format, use_bias=False)
        bn = BatchNormalization(axis = self.bn_axis, momentum=self.bn_momentum)
        lrelu = LeakyReLU(alpha=0.1)
        # self.ds_layers.append([conv, bn, lrelu])
        return [conv, bn, lrelu]

    def create_ResidualBlock(self, filters, block_num, factor = 2):
    # Create Residual Block which performs Convolution with 1x1 Kernel, followed by
    # Convolution with 3x3 Kernel
    # Input before Convolution is added to Output after Convolution
        conv1 = Conv2D(filters=filters, kernel_size=1, strides=1, padding='valid', data_format=self.data_format, use_bias=False)
        bn1 = BatchNormalization(axis = self.bn_axis, momentum=self.bn_momentum)
        lr1 = LeakyReLU(alpha=0.1)
        conv2 = Conv2D(filters=filters * factor, kernel_size=3, strides=1, padding='same', data_format=self.data_format, use_bias=False)
        bn2 = BatchNormalization(axis = self.bn_axis, momentum=self.bn_momentum)
        lr2 = LeakyReLU(alpha=0.1)
        # self.res_layers.append([conv1, bn1, lr1, conv2, bn2, lr2])
        return [conv1, bn1, lr1, conv2, bn2, lr2]

    def create_DropOut(self):
    # Create a Dropout Layer for the Encoder Layer
        drop = Dropout(0.01)
        # self.drop_layers.append(drop)
        return drop

    def create_Decoder_Conv(self, filters, block_num):
    # Create Convolution Layer used before each Residual (Basic) Block
        deconv = Conv2DTranspose(name="Conv2DTranspose", filters=filters, kernel_size=[1, 4], dilation_rate=1, strides=(1, 2), padding='same', data_format=self.data_format, use_bias=False)
        bn = BatchNormalization(axis = self.bn_axis, momentum=self.bn_momentum)
        lrelu = LeakyReLU(alpha=0.1)
        # self.ds_layers.append([conv, bn, lrelu])
        return [deconv, bn, lrelu]

    # def createEncoder(self):
    #     self.count = 0

    #     # Input Layer
    #     self.model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', 
    #         input_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_DEPTH), data_format=self.data_format, use_bias=False))
    #     self.model.add(BatchNormalization(axis = 3, momentum=self.bn_momentum))
    #     self.model.add(LeakyReLU(alpha=0.1))
    #     self.count = 3

    #     for i in range(5):
    #         self.createConv(filters=self.conv_filters[i])
    #         self.createResidual(filters=self.res_filters[i], n_reps=self.residual_reps[i])
    #         self.model.add(Dropout(0.01))
    #         self.count+=1

    #     self.model.compile()
    #     self.model.build()
    #     self.model.summary()
    #     print("Count: ", self.count)
    #     return self.model

    # def define_skip_model(self):  
    #     input_net = Input((32,32,3))
        
    #     ## Encoder starts
    #     conv1 = Conv2D(32, 3, strides=(2,2), activation = 'relu', padding = 'same')(input_net)
    #     conv2 = Conv2D(64, 3, strides=(2,2), activation = 'relu', padding = 'same')(conv1)
    #     conv3 = Conv2D(128, 3, strides=(2,2), activation = 'relu', padding = 'same')(conv2)
        
    #     conv4 = Conv2D(128, 3, strides=(2,2), activation = 'relu', padding = 'same')(conv3)
        
    #     ## And now the decoder
    #     up1 = Conv2D(128, 3, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv4))
    #     merge1 = concatenate([conv3,up1], axis = 3)
    #     up2 = Conv2D(64, 3, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(merge1))
    #     merge2 = concatenate([conv2,up2], axis = 3)
    #     up3 = Conv2D(32, 3, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(merge2))
    #     merge3 = concatenate([conv1,up3], axis = 3)
        
    #     up4 = Conv2D(32, 3, padding = 'same')(UpSampling2D(size = (2,2))(merge3))
        
    #     output_net = Conv2D(3, 3, padding = 'same')(up4)
        
    #     model = Model(inputs = input_net, outputs = output_net)
        
    #     return model