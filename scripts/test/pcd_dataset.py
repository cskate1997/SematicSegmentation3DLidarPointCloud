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
        
        self.yaml = self.read_yaml(yaml_path)
        self.dataset = self.initialize_dataset(dataset_path)
        self.learning_map_inv = self.yaml['learning_map_inv']
        self.learning_map = self.yaml['learning_map']
        # print(type(self.learning_map_inv))

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

    def parse_function(self, filename):
        filename, labels = filename[0], filename[1]
        filename = bytes.decode(filename)
        
        labels = bytes.decode(labels)
        

        points = read_bin_as_array(filename)
        image, labels = self.convert_pcd_image(points, labels)
        
        return [image, labels]

    def initialize_dataset(self, path):

        
        dirs = os.listdir(path)
        if 'sequences' not in dirs:
            print("Sequences folder not found in dataset path, please check the path")
            return
        print("Starting to convert point clouds in range images")

        ind = dirs.index('sequences')
        path = os.path.join(path, dirs[ind])
        path = os.path.abspath(path)
        print("Path = ", path)
        dirs = os.listdir(path)
        dirs.sort()
        print("Dirs = ", dirs)

        filenames = []
        labels = []
        
        for dir in dirs:
            print(f"Converting Sequence {dir}...")
            if dir == '00':
                continue
            if dir == '08':
                break
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
                    filenames.append([tf.constant(path_to_file), tf.constant(path_to_file_label)])
                    labels.append(path_to_file_label)
            else:
                print("no labels")
            # break
        print("Length of dataset = ", len(filenames))
        self.dataset_len = len(filenames)
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(lambda x: tf.py_func(self.parse_function, [x], [tf.double, tf.double]))
        return dataset

    def get_dataset_len(self):
        return self.dataset_len

    def get_dataset(self):
        return self.dataset
    
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