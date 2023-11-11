import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from numpy.core.fromnumeric import reshape

image_height, image_width = 16, 8
temp_image = np.ones((image_height,image_width))
print("Single Channel Image Shape: ", temp_image.shape)

image_channels = 1024
image = np.ones((image_channels,image_height,image_width))

for i in range(0,image_channels):
    image[i,:,:] = (i+1)*temp_image

print("Image Shape: ", image.shape)

scale = 2

output_channels = int (image_channels / (scale*scale))
output_height = scale*image_height
output_width = scale*image_width
print("Output Channel :", output_channels)
print("Output Height :", output_height)
print("Output Width :", output_width)
# while True: pass

pixel_shuffle = np.ones((output_channels, output_height, output_width))

# for c in range(0, output_channels):
for i in range(0, image_height):
    for j in range(0, image_width):            
        temp = np.reshape(image[:,i,j], (output_channels,scale,scale))
        pixel_shuffle[:, scale*i:scale*i+scale, scale*j:scale*j+scale] = temp
            
colormap = colors.ListedColormap(["red","#39FF14","blue","yellow"])
figure = plt.figure(figsize=(output_width,output_height))
plt.imshow(pixel_shuffle[0],cmap=colormap)
plt.show()

