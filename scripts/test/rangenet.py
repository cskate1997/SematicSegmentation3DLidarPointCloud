import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Softmax, Input
from tensorflow.keras.models import Model

from segmentation_head import SegmentationHead
from encoder import Encoder
from decoder import Decoder

INPUT_DEPTH = 5
INPUT_HEIGHT = 64
INPUT_WIDTH = 1024

class RangeNetModel(Model):
    def __init__(self, inp_shape=(None, 64, 1024, 5)):
        super(RangeNetModel, self).__init__()
        # self.inp = Input(shape=inp_shape)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.semantic_head = SegmentationHead()
        self.softmax = Softmax(axis=3)


    def call(self, x):
        # y = self.inp(x)
        
        # print("Input_Shape: ", x.shape)
        # print(x, type(x),x[0])
        # for i in x:
        #     print(type(i), i, i[0])
        
        y = self.encoder(x)
        skips, os = self.encoder.get_skips()
        for i in skips:
            print("==============================")
            print(skips[i].shape)

        self.decoder.set_skips(skips, os)
        y = self.decoder(y)
        y = self.semantic_head(y)
        y = self.softmax(y)
        return y
    
    def summary(self):
        super(RangeNetModel, self).summary()
        count = 24
        print("\n\n"+"="*count+" Encoder Summary "+"="*count+"\n\n")
        self.encoder.summary()
        print("\n\n"+"="*count+" Decoder Summary "+"="*count+"\n\n")
        self.decoder.summary()
        print("\n\n"+"="*count+" Semantic Head Summary "+"="*count+"\n\n")
        self.semantic_head.summary()

if __name__ == '__main__':
    tf.enable_eager_execution()

    # fileName = "model_summary.txt"
    # sys.stdout = open(fileName, "w")

    range_net_model = RangeNetModel()

    range_net_model.build(input_shape=(None, 64, 1024, 5))
    range_net_model.call(Input(shape=(64, 1024, 5)))
    
    range_net_model.summary()