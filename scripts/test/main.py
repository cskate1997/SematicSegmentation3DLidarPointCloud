import sys
from rangenet import RangeNetModel
from pcd_dataset import Dataset
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy

# class CategoricalCrossentropy(tf.keras.losses.Loss):
#     def call(self, y_true, y_pred):
#         return categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)

BATCH_SIZE = 1

if __name__ == "__main__":
    tf.enable_eager_execution()
    if (len(sys.argv) != 3):
        print("Incorrect Number of Arguments provided.\nValid format: \"python file.py -d {path_to_dataset}\"")
        sys.exit(-1)
    print("!!!")
    ind = sys.argv.index("-d")
    path = sys.argv[ind + 1]
    ds = Dataset(path, 'config/semantic-kitti.yaml')
    dataset = ds.get_dataset()
    
    dataset = dataset.batch(BATCH_SIZE)
    data_iter = dataset.make_one_shot_iterator()
    # dataset = dataset.prefetch(1)
    print("Dataset Ready")
    
    # for item in dataset:
    #     print(type(item[0]), item[0].shape)
    #     print(item)
    #     image = item[0].numpy()
    #     print(type(image))
    #     plt.imshow(image[4,:,:], cmap='gray')
    #     plt.show()
    #     break

    range_net_model = RangeNetModel()

    optimizer = SGD(lr=0.005, momentum=0.9, decay=0.0001)
    # loss = CategoricalCrossentropy()
    range_net_model.compile(loss=categorical_crossentropy,
             optimizer=tf.train.AdamOptimizer(learning_rate=0.005),
            #  metrics=[tf.keras.metrics.RootMeanSquaredError()])
             metrics=['accuracy'])

    range_net_model.fit(data_iter, batch_size=BATCH_SIZE, epochs=1, steps_per_epoch=ds.get_dataset_len()//BATCH_SIZE)