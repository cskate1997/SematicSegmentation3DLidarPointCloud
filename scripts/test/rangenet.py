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
    def __init__(self):
        super(RangeNetModel, self).__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.semantic_head = SegmentationHead()
        self.softmax = Softmax(axis=1)


    def call(self, x):
        y = self.encoder(x)
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
    fileName = "model_summary.txt"
    sys.stdout = open(fileName, "w")

    range_net_model = RangeNetModel()

    range_net_model.build(input_shape=(None, 5, 64, 1024))
    range_net_model.call(Input(shape=(5, 64, 1024)))
    range_net_model.summary()