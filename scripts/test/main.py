# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
from rangenet import RangeNetModel
from pcd_dataset import Dataset
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
# from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.layers import Input
import time

BATCH_SIZE = 1
NUM_EPOCHS = 5

def test_visualize():
    for step in test_dataset:
        # print(step[0].shape, step[1].shape)
        ip = step[0]
        y = step[1]
        y_hat = range_net_model(ip)
        y_hat = tf.argmax(y_hat, axis=3)
        y_hat = y_hat.numpy()
        y_hat = y_hat.reshape(64,1024)
        y = tf.argmax(y, axis=3)
        ds.show(y_hat, y.numpy().reshape(64,1024))
        # print(y_hat.shape)
        # break

def test():
    range_net_model.evaluate(test_dataset)

def train():
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='outputs/rangenet.weights.checkpoint-'+time.strftime("%Y_%m_%d-%H:%M:%S")+'.hdf5', save_weights_only=True, verbose=1,save_best_only=True)

    range_net_model.fit(data_iter, epochs=NUM_EPOCHS, steps_per_epoch=ds.get_dataset_len()//BATCH_SIZE, 
                        validation_data=valid_dataset, validation_steps=5, callbacks=[checkpointer])
    timestr = time.strftime("%Y_%m_%d-%H:%M:%S")
    range_net_model.save_weights('outputs/rangenet.weights-'+timestr+'.hdf5')

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
    test_dataset = test_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.prefetch(1)
    print("="*24,"Dataset Ready","="*24)
    
    # for item in dataset:
    #     print(type(item[0]), item[0].shape)
    #     print(item)
    #     image = item[0].numpy()
    #     print(type(image))
    #     plt.imshow(image[4,:,:], cmap='gray')
    #     plt.show()
    #     break
    # if num_gpu > 1:
    #     with tf.device(cpu[0].name):
    #         range_net_model_single = RangeNetModel()
    #         inputs = Input(shape=(64, 1024, 5))
    #         outputs = range_net_model_single(inputs)
    #         model = tf.keras.Model(inputs=inputs, outputs=outputs)
    #         range_net_model_single.summary()
    #         model.summary()
    #     range_net_model = multi_gpu_model(model, gpus=num_gpu)
    # else:
    #     range_net_model = RangeNetModel()

    range_net_model_single = RangeNetModel(rnn_flag=True)
    inputs = Input(shape=(64, 1024, 5))
    outputs = range_net_model_single(inputs)
    range_net_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = SGD(learning_rate=0.005, momentum=0.9, decay=0.0001)
    # loss = CategoricalCrossentropy()
    range_net_model.compile(loss=categorical_crossentropy,
            optimizer=optimizer,
            #  metrics=[tf.keras.metrics.RootMeanSquaredError()])
            metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=20)], run_eagerly=True)

    range_net_model.load_weights('outputs/rnn_b1e5_best.hdf5')

    # train()
    # test()
    test_visualize()