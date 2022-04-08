# import pcl
import sys
import os
from math import radians, pi
import numpy as np
from read_convert_save_view import read_bin_as_array
from matplotlib import pyplot as plt


FOV_DOWN = -24.8
FOV_UP = 2
FOV = abs(FOV_UP) + abs(FOV_DOWN)

HEIGHT = 64
WIDTH = 1024

def convert_pcd_image(points):
    spherical_image = np.zeros((HEIGHT, WIDTH, 5))
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

    return spherical_image



def save_image(image, path, binName):
    fileName = binName.split('.')[0]
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, fileName)
    plt.imsave(path + "_x.jpg", image[:,:,0], cmap='gray')
    plt.imsave(path + "_y.jpg", image[:,:,1], cmap='gray')
    plt.imsave(path + "_z.jpg", image[:,:,2], cmap='gray')
    plt.imsave(path + "_r.jpg", image[:,:,3], cmap='gray')
    plt.imsave(path + "_i.jpg", image[:,:,4], cmap='gray')
    # pass


def main(argv):
    if (len(argv) != 3):
        print("Incorrect Number of Arguments provided.\nValid format: \"python file.py -d {path_to_dataset}\"")
        return
    ind = argv.index("-d")
    path = argv[ind + 1]
    dirs = os.listdir(path)
    if 'sequences' not in dirs:
        print("Sequences folder not found in dataset path, please check the path")
        return
    ind = dirs.index('sequences')
    path = os.path.join(path, dirs[ind])
    path = os.path.abspath(path)
    # print(path)
    dirs = os.listdir(path)
    dirs.sort()
    # print(dirs)
    for dir in dirs:
        path_to_seq = os.path.join(path, dir)
        path_to_velo= os.path.join(path_to_seq, 'velodyne')
        # print(path_to_velo)
        files = os.listdir(path_to_velo)
        files.sort()
        for file in files:
            print(dir, file)
            path_to_file = os.path.join(path_to_velo, file)
            points = read_bin_as_array(path_to_file)
            # cloud = convert_to_point_cloud(points, "XYZI")
            image = convert_pcd_image(points)
            # path_to_image = os.path.join(path_to_seq, 'images')
            # save_image(image, path_to_image, file)
            # plt.imshow(image[:,:,4], cmap='gray')
            # plt.show(block=False)
            # plt.pause(0.0003)
            # return
            # print(path_to_file)
    
if __name__ == '__main__':
    main(sys.argv)
