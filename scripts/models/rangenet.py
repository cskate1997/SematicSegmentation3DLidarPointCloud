import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Softmax, Input
from tensorflow.keras.models import Model

from models.segmentation_head import SegmentationHead
from models.encoder import Encoder
from models.decoder import Decoder

INPUT_DEPTH = 5
INPUT_HEIGHT = 64
INPUT_WIDTH = 1024

class RangeNetModel(Model):
    def __init__(self, inp_shape=(None, 64, 1024, 5), rnn_flag=False, pixel_shuffle=False):
        super(RangeNetModel, self).__init__()
        # self.inp = Input(shape=inp_shape)
        self.encoder = Encoder(pixel_shuffle=pixel_shuffle)
        self.decoder = Decoder(pixel_shuffle=pixel_shuffle)
        self.semantic_head = SegmentationHead()
        self.softmax = Softmax(axis=3)
        self.rnn_flag = rnn_flag


    def call(self, x):        
        y = self.encoder(x)
        
        skips, os = self.encoder.get_skips()

        if self.rnn_flag:
            flag = False
            try:
                cur_y = y.numpy()
                flag = True
            except:
                pass
            if flag:
                if hasattr(self, 'prev_y'):
                    y = y + tf.constant(self.prev_y)
                self.prev_y = cur_y

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
    # tf.enable_eager_execution()

    # fileName = "outputs/model_summary_pxl.txt"
    # sys.stdout = open(fileName, "w")

    range_net_model = RangeNetModel(rnn_flag=False, pixel_shuffle=False)

    range_net_model.build(input_shape=(None, 64, 1024, 5))
    range_net_model.call(Input(shape=(64, 1024, 5)))
    
    range_net_model.summary()