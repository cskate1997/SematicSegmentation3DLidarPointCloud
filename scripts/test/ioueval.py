import numpy as np
import tensorflow as tf
import sys

from CustomIoU import *

'''
Reference: - 
https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
'''

class iouEval:
    def __init__(self, num_classes, ignore_list):
        self.num_classes = num_classes
        self.ignore = tf.Tensor(ignore_list)
        # Convert self.ignore tensor to long PENDING
        self.include = tf.Tensor([i for i in range(self.ignore) if i not in self.ignore])
        # Convert self.include tensor to long PENDING
        self.confusion_matrix = tf.fill(dims=[self.num_classes, self.num_classes], value=0)
        self.ones = None
        self.scan_size = None

    def addCompareData(self, ground_truth, prediction):
        if isinstance(ground_truth, np.ndarray):
            tf.convert_to_tensor(value = ground_truth, dtype = np.int64)
        if isinstance(prediction, np.ndarray):
            tf.convert_to_tensor(value = prediction, dtype = np.int64)
        yhat = prediction.reshape(-1)
        y = ground_truth.reshape(-1)
        indexes = tf.stack(values=[yhat, y], axis=0)
        if ((self.ones is None) or (self.scan_size != indexes.shape[-1])):
            self.ones = tf.fill(dims=indexes.shape[-1], value=1)
            # Convert self.ones tensor to long PENDING
            self.scan_size = indexes.shape[-1]

        # x - ground truth
        # y - predictions
        self.confusion_matrix = tf.gather_nd(params=self.ones, indices=tuple(indexes), dims=-1)
        # self.confusion_matrix = tf.gfn = ather(params=self.ones, indices=tuple(indexes), dims=-1)

    def getProbabilities(self):
        confusion_matrix = np.copy(self.confusion_matrix)
        confusion_matrix = confusion_matrix.astype(np.double)
        confusion_matrix[self.ignore] = 0
        confusion_matrix[:,self.ignore] = 0

        true_positives = np.diag(confusion_matrix)
        false_positives = np.sum(confusion_matrix, axis = 1) - true_positives
        false_negatives = np.sum(confusion_matrix, axis = 0) - true_positives
        true_negatives = np.sum(confusion_matrix) - (true_positives + false_positives + false_negatives)
        return true_positives, true_negatives, false_positives, false_negatives

    def getIoU(self):
        tp, tn, fp, fn = self.getProbabilities()
        intersection = tp
        union = tp + fp + fn
        iou_score = intersection / union
        iou_mean = np.mean(intersection([self.include]) / union([self.include]))
        return iou_score

    def getAccuracy(self):
        tp, tn, fp, fn = self.getProbabilities()
        accuracy = np.sum(tp) / (np.sum(tp[self.include]) + np.sum(tn[self.include]))
        return accuracy

def create_confusion_matrix(truth, prediction):
    truth = truth.flatten()
    print("Truth :\n", truth)
    prediction = prediction.flatten()
    print("Prediction :\n", prediction)
    classes = np.unique(truth)
    print("Classes :\n", len(classes), classes)
    confusion_matrix = np.zeros((len(classes), len(classes)))
    for i in range(len(classes)):
        for j in range(len(classes)):
            # print("i : ", i, " j :",j)
            confusion_matrix[i,j] = np.sum( (truth == classes[i]) & (prediction == classes[j]) )
            # print(confusion_matrix)
    return confusion_matrix

if __name__ == "__main__":
    sys.stdout
    # a = np.random.randint(0,10,(64,1024))
    a = np.zeros((10,10))
    a[:10,:5] = 1
    print("a :\n", a)
    # b = np.random.randint(0,10,(64,1024))
    b = np.zeros((10,10))
    b[:5, :5] = 1
    print("b :\n", b)

    conf = create_confusion_matrix(a,b)
  
    print("Confusion Matrix:\n", conf)
    true_positives = np.diag(conf)
    true_positives = np.atleast_2d(true_positives).T
    print("True+ve\n", true_positives)
    false_positives = np.sum(conf, axis = 1, keepdims=True) - true_positives
    print("False+ve\n", false_positives)
    false_negatives = np.sum(conf, axis = 0, keepdims=True).T - true_positives
    print("False-ve\n", false_negatives)
    true_negatives = np.sum(conf) - (true_positives + false_positives + false_negatives)
    print("True-ve\n", true_negatives)
    intersection = true_positives
    print("Intersection\n", intersection)
    union = true_positives + false_positives + false_negatives
    print("Union\n", union)
    iou_score = intersection / union
    print("IoU\n", iou_score)
    iou_mean = np.mean(iou_score)
    print("IoU Mean\n", iou_mean)    

    whatever = CustomIoU(classes=2)
    whatever.update_state(a,b)
    print("IOU TF", whatever.result())