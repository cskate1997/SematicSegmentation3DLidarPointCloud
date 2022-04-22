from email import message
import tensorflow as tf
import sys
import os
from math import radians, pi
import numpy as np
from read_convert_save_view import read_bin_as_array, read_labels
from matplotlib import pyplot as plt
import yaml

FOV_DOWN = -24.8
FOV_UP = 2
FOV = abs(FOV_UP) + abs(FOV_DOWN)

HEIGHT = 64
WIDTH = 1024

class Dataset:
    def __init__ (self, dataset_path, yaml_path):
        self.dataset_path = dataset_path
        self.yaml_path = yaml_path
        self.yaml = self.read_yaml(yaml_path)
        # self.dataset = self.initialize_dataset(dataset_path)
        self.learning_map_inv = self.yaml['learning_map_inv']
        self.learning_map = self.yaml['learning_map']
        self.sequences = self.get_sequences()
        # print(type(self.learning_map_inv))
        self.plt1 = plt.figure()
        self.plt2 = plt.figure()
        self.ax1 = self.plt1.add_subplot(111)
        self.ax2 = self.plt2.add_subplot(111)
        

    def label_mapper(self, x):
        return self.learning_map[x]

    def label_map_inverter(self, x):
        return self.learning_map_inv[x]

    def read_yaml(self, path):
        with open(path, "r") as stream:
            try:
                yamlFile = yaml.safe_load(stream)
                # print(yamlFile)
            except yaml.YAMLError as exc:
                print(exc)
        # colors = yamlFile['color_map']
        return yamlFile

    def convert_pcd_image(self, points, labels):
        labels = read_labels(labels)
        labels = np.vectorize(self.label_mapper)(labels)
        spherical_image = np.zeros((HEIGHT, WIDTH, 5))
        labels_image = np.zeros((HEIGHT, WIDTH, 20))
        Rs = np.linalg.norm(points, axis=1)
        pitches = points[:,2]
        pitches = pitches / Rs
        pitches = np.arcsin(pitches)
        yaws = np.arctan2(points[:,1], points[:,0])

        pitches =  (HEIGHT - 1) * (1 - (pitches + abs(radians(FOV_DOWN))) / radians(FOV))
        yaws = (WIDTH - 1) * (0.5 * ((yaws / pi) + 1))
        pitches = np.clip(pitches, 0 , (HEIGHT - 1))
        yaws = np.clip(yaws, 0 , (WIDTH - 1))

        pitches = np.round(pitches).astype(int)
        yaws = np.round(yaws).astype(int)

        spherical_image[pitches, yaws, 0] = points[:,0]
        spherical_image[pitches, yaws, 1] = points[:,1]
        spherical_image[pitches, yaws, 2] = points[:,2]
        spherical_image[pitches, yaws, 3] = Rs
        spherical_image[pitches, yaws, 4] = points[:,3]
        labels_image[pitches, yaws, labels] = 1

        # spherical_image = np.reshape(spherical_image, (1, WIDTH*HEIGHT, 5))
        return spherical_image, labels_image

    def show(self, img, ip):
        height, width = img.shape
        rgb_img = np.zeros((height, width, 3))
        rgb_ip = np.zeros((height, width, 3))
        color_map = self.yaml['color_map']
        for i in range(height):
            for j in range(width):
                # print(color_map[self.learning_map_inv[img[i, j]]] , color_map[self.learning_map_inv[ip[i, j]]])
                rgb_img[i,j,:] = color_map[self.learning_map_inv[img[i, j]]]
                rgb_ip[i,j,:] = color_map[self.learning_map_inv[ip[i, j]]]
        # print("MaxVals", np.max(rgb_img), np.max(rgb_ip), np.min(rgb_img), np.min(rgb_ip))
        
        # self.plt1.imshow(rgb_img)
        self.im1 = self.ax1.imshow(rgb_img)
        self.im2 = self.ax2.imshow(rgb_ip)
        # self.plt2.imshow(rgb_ip)
        # plt.show(block=False)
        # plt.pause(0.03)
        self.plt1.canvas.draw()
        self.plt2.canvas.draw()
        self.plt1.canvas.flush_events()
        self.plt2.canvas.flush_events()
        plt.show(block=False)


    def parse_function(self, filename):
        filename, labels = filename[0], filename[1]
        # print("="*20+"Filename, Label: ", filename, labels)
        filename = filename.numpy()
        
        labels = labels.numpy()
        # print("="*20+"Filename, Label: ", filename, labels)

        points = read_bin_as_array(filename)
        image, labels = self.convert_pcd_image(points, labels)
        
        return [image, labels]

    def get_sequences(self):
        path = self.dataset_path
        if not hasattr(self, 'sequences'):
            print("Sequence is none, getting sequences")
            dirs = os.listdir(path)
            if 'sequences' not in dirs:
                print("Sequences folder not found in dataset path, please check the path")
                return
            # print("Starting to convert point clouds in range images")

            ind = dirs.index('sequences')
            path = os.path.join(path, dirs[ind])
            path = os.path.abspath(path)
            self.sequence_path = path
            # print("Path = ", path)
            dirs = os.listdir(path)
            dirs.sort()
            return dirs
        else:
            return self.sequences

    def init_train_data(self, dirs, shuffle=True):
        print("Initializing Training dataset")
        self.train_dirs = dirs
        self.train_dataset, self.train_len = self.initialize_dataset(dirs, shuffle)

    def init_valid_data(self, dirs, shuffle=True):
        print("Initializing Validation dataset")
        self.valid_dirs = dirs
        self.valid_dataset, self.valid_len = self.initialize_dataset(dirs, shuffle)

    def init_test_data(self, dirs, shuffle=False):
        self.test_dirs = dirs
        print("Initializing Testing dataset")
        self.test_dataset, self.test_len = self.initialize_dataset(dirs, shuffle)

    def get_train_data(self):
        if not hasattr(self, 'train_dataset'):
            print("Training Dataset is not initialized")
            return None
        else:
            return self.train_dataset

    def get_valid_data(self):
        if not hasattr(self, 'valid_dataset'):
            print("Validation Dataset is not initialized")
            return None
        else:
            return self.valid_dataset

    def get_test_data(self):
        if not hasattr(self, 'test_dataset'):
            print("Testing Dataset is not initialized")
            return None
        else:
            return self.test_dataset

    def get_dataset_len(self, typ='train'):
        if typ == 'train':
            return self.train_len
        if typ == 'valid':
            return self.valid_len
        if typ == 'test':
            return self.test_len

    def initialize_dataset(self, dirs, shuffle=False):   
        path = self.sequence_path
        print("Dirs = ", dirs)

        filenames = []
        labels = []
        
        for dir in dirs:
            print(f"Converting Sequence {dir}...")
            path_to_seq = os.path.join(path, dir)
            path_to_labels = os.path.join(path_to_seq, 'labels')
            path_to_velo= os.path.join(path_to_seq, 'velodyne')

            # print(path_to_velo)
            
            if os.path.isdir(path_to_labels):
                files = os.listdir(path_to_velo)
                files.sort()
                # images = np.zeros((len(files), 5, HEIGHT, WIDTH))
                i = 0
                for file in files:
                    path_to_file = os.path.join(path_to_velo, file)
                    
                    path_to_file_label = os.path.join(path_to_labels, file.split(".")[0]+".label")
                    filenames.append([path_to_file, path_to_file_label])
                    labels.append(path_to_file_label)
            else:
                print("no labels")
            # break
        print("Length of dataset = ", len(filenames))
        dataset_len = len(filenames)
        if shuffle:
            np.random.shuffle(filenames)
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(lambda x: tf.py_function(self.parse_function, [x], [tf.float32, tf.float32]))

        return dataset, dataset_len


    
if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print("Incorrect Number of Arguments provided.\nValid format: \"python file.py -d {path_to_dataset}\"")
        sys.exit(-1)
    print("!!!")
    ind = sys.argv.index("-d")
    path = sys.argv[ind + 1]
    ds = Dataset(path, 'config/semantic-kitti.yaml')
    dataset = ds.get_dataset()
    for one_element in dataset:
        print(one_element[1], one_element[1].shape)
        break

    print("Hello, World!")