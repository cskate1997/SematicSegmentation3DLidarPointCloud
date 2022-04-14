
import numpy as np
import matplotlib.pyplot as plt
import os, sys

print("Hello User!")

dataset = []
size = []
total_sum = 0
for i in range(0,1):
    # print(f'Data {i:02}')
    dataset.append(np.load(f'/work/barane/data_odometry_velodyne/dataset/sequences/{i:02}/images/projections.npz'))
    size.append(np.shape((dataset[i])['arr_0']))
    print(f'Shape {i:02}', size[i])
    total_sum += (size[i])[0]
print(f'Total {i:02}', total_sum)

'''
x:1, y:2, z:3, r:4, i:5
data[sequence, sequence_size, image]
'''
data = np.array([])

for sequence in range(0,1):
    for sequence_size in range(0, (size[sequence])[0]):
        image = (((dataset[sequence])['arr_0'])[sequence_size]).reshape(64,1024,5)
print("Image Shape :", np.shape(image))

        # for image_type in range(0,5):
        #     data[sequence, sequence_size, image_type] = image[:,:,image_type]

print("Total Shape =", np.shape(data))
print("Shape = ", np.shape(data[0,1,2]))

scan_files = []
label_files = []