# import pcl
from cmath import pi
from math import asin, atan2, radians
import numpy as np
import pcl
from read_convert_save_view import read_bin_as_array, convert_to_point_cloud, visualize_point_cloud
from matplotlib import pyplot as plt


FOV_DOWN = -24.8
FOV_UP = 2
FOV = abs(FOV_UP) + abs(FOV_DOWN)

points = read_bin_as_array("data/sequences/00/velodyne/000000.bin")
cloud = convert_to_point_cloud(points, "XYZI")
# visualize_point_cloud(cloud, "XYZI")

spherical_image = np.zeros((64, 1024, 6))

total = points.shape[0]
count = 0
pitches = []
yaws = []
for point in points:
    print("Converting point cloud to spherical image... "+str(int(count/total*100))+"% done.",end="\r")
    count += 1
    R = (point[0]**2 + point[1]**2 + point[2]**2) ** 0.5
    pitch = asin(point[2]/ R)
    yaw = atan2(point[1], point[0])
    # pitch = (1 - (pitch + FOV_DOWN)) / FOV
    # yaw = (yaw + pi)/(2*pi)
    pitch =  63 * (1 - (pitch + abs(radians(FOV_DOWN))) / radians(FOV))
    yaw = 1023 * (0.5 * ((yaw / pi) + 1))

    pitch = int(np.round(max(min(pitch, 63),0)))
    yaw = int(np.round(max(min(yaw, 1023),0)))
    # pitches.append(pitch)
    # yaws.append(yaw)
    # pitches.append(pitch)
    # yaws.append(yaw)
    # print(pitch, yaw)
    spherical_image[pitch,yaw,0] = point[0]
    spherical_image[pitch,yaw,1] = point[1]
    spherical_image[pitch,yaw,2] = point[2]
    spherical_image[pitch,yaw,3] = R
    spherical_image[pitch,yaw,4] = point[3]

# print()
# print(max(pitches), min(pitches), max(yaws), min(yaws))

# plt.imshow(spherical_image[:,:,0], cmap='gray')
# plt.show()
# plt.imshow(spherical_image[:,:,1], cmap='gray')
# plt.show()
# plt.imshow(spherical_image[:,:,2], cmap='gray')
# plt.show()
# plt.imshow(spherical_image[:,:,3], cmap='gray')
# plt.show()
# plt.imshow(spherical_image[:,:,4], cmap='gray')
# plt.show()
# plt.imshow(spherical_image[:,:,3:6])
# plt.show()

plt.imsave("data/sequences/00/images/X_channel/000000.jpg", spherical_image[:,:,0], cmap='gray')
plt.imsave("data/sequences/00/images/Y_channel/000000.jpg", spherical_image[:,:,1], cmap='gray')
plt.imsave("data/sequences/00/images/Z_channel/000000.jpg", spherical_image[:,:,2], cmap='gray')
plt.imsave("data/sequences/00/images/R_channel/000000.jpg", spherical_image[:,:,3], cmap='gray')
plt.imsave("data/sequences/00/images/I_channel/000000.jpg", spherical_image[:,:,4], cmap='gray')
