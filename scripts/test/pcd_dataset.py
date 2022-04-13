from email import message
import tensorflow as tf
import sys
import os
from math import radians, pi
import numpy as np
from read_convert_save_view import read_bin_as_array, read_labels
from matplotlib import pyplot as plt
from tensorflow.data import Dataset

FOV_DOWN = -24.8
FOV_UP = 2
FOV = abs(FOV_UP) + abs(FOV_DOWN)

HEIGHT = 64
WIDTH = 1024

def convert_pcd_image(points):
    spherical_image = np.zeros((5, HEIGHT, WIDTH))
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

    spherical_image[0, pitches, yaws] = points[:,0]
    spherical_image[1, pitches, yaws] = points[:,1]
    spherical_image[2, pitches, yaws] = points[:,2]
    spherical_image[3, pitches, yaws] = Rs
    spherical_image[4, pitches, yaws] = points[:,3]

    # spherical_image = np.reshape(spherical_image, (1, WIDTH*HEIGHT, 5))
    return spherical_image

def convert_pcd_image(points, labels):
    labels = read_labels(labels)
    spherical_image = np.zeros((5, HEIGHT, WIDTH))
    labels_image = np.zeros((1, HEIGHT, WIDTH))
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

    spherical_image[0, pitches, yaws] = points[:,0]
    spherical_image[1, pitches, yaws] = points[:,1]
    spherical_image[2, pitches, yaws] = points[:,2]
    spherical_image[3, pitches, yaws] = Rs
    spherical_image[4, pitches, yaws] = points[:,3]
    labels_image[0, pitches, yaws] = labels

    # spherical_image = np.reshape(spherical_image, (1, WIDTH*HEIGHT, 5))
    return spherical_image, labels_image

'''
def save_image(image, path):
    fileName = "projections"
    if not os.path.isdir(path):
        os.makedirs(path)
    else:
        print("deleting directory")
        shutil.rmtree(path)
        os.makedirs(path)
    path = os.path.join(path, fileName)
    np.savez_compressed(path, image)
    plt.imsave(path + "_x.jpg", image[:,:,0], cmap='gray')
    plt.imsave(path + "_y.jpg", image[:,:,1], cmap='gray')
    plt.imsave(path + "_z.jpg", image[:,:,2], cmap='gray')
    plt.imsave(path + "_r.jpg", image[:,:,3], cmap='gray')
    plt.imsave(path + "_i.jpg", image[:,:,4], cmap='gray')
    pass
'''
tf.enable_eager_execution()


def parse_function(filename):
    # print(filename)
    # filename = tf.io.parse_tensor(filename, tf.string)
    # filename = tf.convert_to_tensor(filename)
    filename, labels = filename[0], filename[1]
    # print("=========", filename.numpy())
    # filename = tf.read_file(filename)
    # print(type(filename))
    # filename = tf.string.
    # print(filename)
    # print("=====================", filename, labels)
    filename = bytes.decode(filename)
    labels = bytes.decode(labels)
    # with tf.Session() as sess:
    #     tf.Print(filename, [filename], message="Hello: ")
    # print("=====================", filename, labels, type(filename), type(labels))
    # filename = tf.Print(filename)
    # label = tf.io.read_file(label)
    points = read_bin_as_array(filename)
    # print("=====================", filename, labels, type(filename), type(labels))
    # labels = read_labels(labels)
    image, labels = convert_pcd_image(points, labels)
    # print("Image Shape, Labels Shape: ",image.shape, labels.shape)
    return [image, labels]

def getDataset(path):

    
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
        # if dir == '00':
        #     continue
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
                # break
                # print(dir, path_to_file)
                # points = read_bin_as_array(path_to_file)
                # cloud = convert_to_point_cloud(points, "XYZI")
                # image = convert_pcd_image(points)
                # images[i,:,:,:] =image
                # i+=1
            # filenames = np.concatenate((filenames, images), axis=0)
        else:
            print("no labels")
    print("Length of dataset = ", len(filenames))
    # print(filenames[:1])
    # print(labels[:5])
    # print(tf.executing_eagerly())
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    # print(dataset)
    dataset = dataset.map(lambda x: tf.py_func(parse_function, [x], [tf.double, tf.double]))
    return dataset
    # for one_element in dataset:
    #     print(type(one_element))

    # dataset = dataset.map(parse_function)
    
    

    # dataset = dataset.batch(5)
    # dataset = dataset.prefetch(1)
    # iterator = dataset.make_initializable_iterator()
    # next_element = iterator.get_next()
    # with tf.Session() as sess:
    #     print(sess.run(next_element))
        # path_to_image = os.path.join(path_to_seq, 'images')
        # save_image(images, path_to_image)
            # plt.imshow(image[:,:,4], cmap='gray')
            # plt.show(block=False)
            # plt.pause(0.0003)
            # return
            # print(path_to_file)
    
if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print("Incorrect Number of Arguments provided.\nValid format: \"python file.py -d {path_to_dataset}\"")
        sys.exit(-1)
    print("!!!")
    ind = sys.argv.index("-d")
    path = sys.argv[ind + 1]
    dataset = getDataset(path)
    for one_element in dataset:
        print(one_element[1], one_element[1].shape)
        break

    print("Hello, World!")