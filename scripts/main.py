# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
from models.rangenet import RangeNetModel
from utils.pcd_dataset import Dataset
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
# from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.layers import Input
import time
import numpy as np
from utils.CustomIoU import CustomIoU
import json

BATCH_SIZE = 2
NUM_EPOCHS = 10

def test_visualize(multiple=False):
    ds.init_plot()
    for step in test_dataset:
        ip = step[0]
        y = step[1]
        y_hat = range_net_model(ip)
        y_hat = tf.argmax(y_hat, axis=3)
        y_hat = y_hat.numpy()
        y_hat = y_hat.reshape(64,1024)

        y = tf.argmax(y, axis=3)
        if multiple:
            y_hat_rnn = range_net_model_2(ip)
            y_hat_rnn = tf.argmax(y_hat_rnn, axis=3)
            y_hat_rnn = y_hat_rnn.numpy()
            y_hat_rnn = y_hat_rnn.reshape(64,1024)
            ds.show(y_hat, y_hat_rnn, y.numpy().reshape(64,1024))
        else:
            ds.show(y_hat, y.numpy().reshape(64,1024))


def test():
    range_net_model.evaluate(test_dataset)

def train():
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='outputs/rangenet.weights.checkpoint-'+time.strftime("%Y_%m_%d-%H:%M:%S")+'.hdf5', save_weights_only=True, verbose=1,save_best_only=True)

    history = range_net_model.fit(data_iter, epochs=NUM_EPOCHS, steps_per_epoch=ds.get_dataset_len()//BATCH_SIZE, 
                        validation_data=valid_dataset, validation_steps=5, callbacks=[checkpointer])
    timestr = time.strftime("%Y_%m_%d-%H:%M:%S")
    json.dump(history.history, open('outputs/rangenet.history-'+timestr+'.json', 'w'))
    
    range_net_model.save_weights('outputs/rangenet.weights-'+timestr+'.hdf5')


def weighted_categorical_crossentropy(weights):
    weights = tf.keras.backend.variable(weights)
    def loss(y_true, y_pred):
        y_true = y_true*weights
        cce = tf.keras.losses.CategoricalCrossentropy(axis=-1)
        loss = cce(y_true, y_pred)
        # y_pred /= tf.keras.backend.sum(y_pred, axis = -1, keepdims=True)
        # y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1-tf.keras.backend.epsilon())
        # loss = y_true * tf.keras.backend.log(y_pred) * weights
        # loss = -tf.keras.backend.sum(loss, -1)
        return loss
    return loss


if __name__ == "__main__":
    # tf.enable_eager_execution()
    if (len(sys.argv) != 3):
        print("Incorrect Number of Arguments provided.\nValid format: \"python file.py -d {path_to_dataset}\"")
        sys.exit(-1)
    # devices = tf.keras.backend.get_session().list_devices()
    # print(devices)
    # cpu = [device for device in devices if device.device_type.lower() == 'cpu']
    # devices = [device for device in devices if device.device_type.lower() == 'gpu']
    # num_gpu = len(devices)
    num_gpu = 1
    # print(cpu)
    print("GPUs Available: ", num_gpu)
    print("="*24,"Initializing Dataset","="*24)
    ind = sys.argv.index("-d")
    path = sys.argv[ind + 1]
    ds = Dataset(path, 'config/semantic-kitti.yaml')
    sequences = ds.get_sequences()
    print("Sequences: ", sequences)
    ds.init_train_data(sequences[1:8])
    print("Training Sequences: ", ds.train_dirs)
    ds.init_valid_data(sequences[8:10])
    print("Validation Sequences: ", ds.valid_dirs)
    ds.init_test_data(sequences[0:1])
    print("Testing Sequences: ", ds.test_dirs)
    dataset = ds.get_train_data()
    valid_dataset = ds.get_valid_data()
    test_dataset = ds.get_test_data()
    dataset = dataset.repeat(NUM_EPOCHS)
    # dataset = dataset.shuffle(100)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)
    data_iter = dataset
    # valid_dataset = valid_dataset.repeat(NUM_EPOCHS)
    # dataset = dataset.shuffle(100)
    valid_dataset = valid_dataset.batch(BATCH_SIZE)
    valid_dataset = valid_dataset.prefetch(1)
    # test_dataset = test_dataset.repeat(NUM_EPOCHS)
    # dataset = dataset.shuffle(100)
    test_dataset = test_dataset.batch(1)
    test_dataset = test_dataset.prefetch(1)
    print("="*24,"Dataset Ready","="*24)


    range_net_model_single = RangeNetModel(rnn_flag=False, pixel_shuffle=True)
    inputs = Input(shape=(64, 1024, 5))
    outputs = range_net_model_single(inputs)
    range_net_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    class_weights = ds.get_class_weights()
    
    
    eval_ignore_list = [0]

    optimizer = SGD(learning_rate=0.005, momentum=0.9, decay=0.0001)
    
    range_net_model.compile(loss=categorical_crossentropy,
        optimizer='adam',
        metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=20),
            # tf.keras.metrics.RootMeanSquaredError(),
            CustomIoU(classes=20, ignore=eval_ignore_list),
            CustomIoU(classes=20, ignore=eval_ignore_list, ind=[1], name='class1'),
            CustomIoU(classes=20, ignore=eval_ignore_list, ind=[2], name='class2'),
            CustomIoU(classes=20, ignore=eval_ignore_list, ind=[3], name='class3'),
            CustomIoU(classes=20, ignore=eval_ignore_list, ind=[4], name='class4'),
            CustomIoU(classes=20, ignore=eval_ignore_list, ind=[5], name='class5'),
            CustomIoU(classes=20, ignore=eval_ignore_list, ind=[6], name='class6'),
            CustomIoU(classes=20, ignore=eval_ignore_list, ind=[7], name='class7'),
            CustomIoU(classes=20, ignore=eval_ignore_list, ind=[8], name='class8'),
            CustomIoU(classes=20, ignore=eval_ignore_list, ind=[9], name='class9'),
            CustomIoU(classes=20, ignore=eval_ignore_list, ind=[10], name='class10'),
            CustomIoU(classes=20, ignore=eval_ignore_list, ind=[11], name='class11'),
            CustomIoU(classes=20, ignore=eval_ignore_list, ind=[12], name='class12'),
            CustomIoU(classes=20, ignore=eval_ignore_list, ind=[13], name='class13'),
            CustomIoU(classes=20, ignore=eval_ignore_list, ind=[14], name='class14'),
            CustomIoU(classes=20, ignore=eval_ignore_list, ind=[15], name='class15'),
            CustomIoU(classes=20, ignore=eval_ignore_list, ind=[16], name='class16'),
            CustomIoU(classes=20, ignore=eval_ignore_list, ind=[17], name='class17'),
            CustomIoU(classes=20, ignore=eval_ignore_list, ind=[18], name='class18'),
            CustomIoU(classes=20, ignore=eval_ignore_list, ind=[19], name='class19'),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
            ],
        run_eagerly=True, 
        loss_weights=class_weights
        )
            # metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=20), tf.keras.metrics.IoU(num_classes=20, target_class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14,15,16,17,18,19])], run_eagerly=True)

    ## FOR TESTING or VISUALIZATION ##
    # range_net_model.load_weights('outputs/rnn_b1e10_srqt_loss_shuffle_off.hdf5') # Change RNN and Pixel Shuffle Flags accordingly!!!

    ## FOR TRAINING ##
    train()
    
    ## FOR TESTING ##
    # test()

    ## FOR VISUALIZATION ##
    # visualize_multiple = False
    # if visualize_multiple:
    #     range_net_model_single_2 = RangeNetModel(rnn_flag=False, pixel_shuffle=True)
    #     inputs = Input(shape=(64, 1024, 5))
    #     outputs = range_net_model_single_2(inputs)
    #     range_net_model_2 = tf.keras.Model(inputs=inputs, outputs=outputs)

    #     optimizer = SGD(learning_rate=0.005, momentum=0.9, decay=0.0001)
    #     # loss = CategoricalCrossentropy()
    #     range_net_model_2.compile(loss=categorical_crossentropy,
    #             optimizer=optimizer,
    #             #  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    #             metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=20)], run_eagerly=True)

    #     range_net_model_2.load_weights('outputs/rangenet_b2e5.hdf5')
    #     test_visualize(multiple=visualize_multiple)
    # else:
    #     test_visualize()
