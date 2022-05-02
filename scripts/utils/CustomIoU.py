import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Metric

class CustomIoU(Metric):
    def __init__(self, name="c_iou", classes=20, ind=None, ignore=[], weights=None, **kwargs):
        super(CustomIoU, self).__init__(name=name, **kwargs)
        self.classes = classes
        self.ignore = tf.constant(ignore)
        self.ind = ind
        self.include = tf.constant([n for n in range(self.classes) if n not in ignore])
        if weights is not None:
            self.weights_cust = tf.constant(weights, dtype=tf.float32)
        else:
            self.weights_cust = None
        self.reset()

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.weights_cust is not None:
            y_pred = y_pred * self.weights_cust
        y_true = tf.argmax(y_true, axis=3) 
        y_pred = tf.argmax(y_pred, axis=3) 
        y_true_row = tf.reshape(y_true, (-1))
        y_pred_row = tf.reshape(y_pred, (-1))
        self.confusion_matrix = self.confusion_matrix + tf.math.confusion_matrix(y_true_row, y_pred_row, num_classes = self.classes, dtype=tf.float32)

    def result(self):
        return self.getIoU()

    def reset(self):
        self.confusion_matrix = tf.keras.backend.variable(tf.zeros((self.classes, self.classes)))
        self.ones = None
        self.last_scan_size = None

    def getStats(self):
        conf = tf.keras.backend.variable(self.confusion_matrix)
        for i in self.ignore:
            conf[i].assign(conf[i]*0)
            conf[:, i].assign(conf[:, i]*0)
        # get the clean stats
        tp = tf.linalg.diag_part(conf)
        fp = tf.math.reduce_sum(conf, axis=1) - tp
        fn = tf.math.reduce_sum(conf, axis=0) - tp
        return tp, fp, fn

    def getIoU(self):
        tp, fp, fn = self.getStats()
        intersection = tp
        union = tp + fp + fn + 1e-15
        inters= tf.gather(intersection, self.include)
        un = tf.gather(union, self.include)
        iou = intersection / union
        iou_mean = tf.math.reduce_mean(inters / un)
        if self.ind is None:
            return iou_mean 
        else:
            return iou[self.ind]