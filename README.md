# Semantic Segmentation of LiDAR Point Cloud for Autonomous Vehicles

### *CS541: Deep Learning - [Worcester Polytechnic Institute](https://www.wpi.edu/), Spring 2022*

### Members: [Chinmay Madhukar Todankar](https://github.com/chinmaytodankar), [Bhushan Ashok Rane](https://github.com/ranebhushan), [Aniket Manish Patil](https://github.com/aniketmpatil)

Master of Science in Robotics Engineering

#### [Link to Report](./CS541_Group7_Final_Project_Report.pdf)

--------------------------------------------------------------

## Requirements:

1. CUDA Toolkit + GPU drivers

2. [Tensorflow](https://www.tensorflow.org/install)

3. Numpy

4. Matplotlib

5. Pillow

--------------------------------------------------------------

## Dataset - Semantic KITTI

Download the Velodyne sensor data and the Label data folders, and place in the `dataset` folder in the form as mentioned on the [Semantic KITTI website](http://www.semantic-kitti.org/dataset.html#overview).

1. [Download](http://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip) Point Cloud Data

2. [Download](http://www.semantic-kitti.org/assets/data_odometry_labels.zip) Label Data

We will require the path of this `dataset` folder as a argument to the run command.

--------------------------------------------------------------

## How to run the code:

Go to the parent folder of this repo, that is, [semantic_segmentation](.) and enter the command:
  ```
  python3 scripts/main.py -d **path_to_dataset_folder**
  ```

----------------------
## References
1. A. Milioto and I. Vizzo and J. Behley and C. Stachniss, [RangeNet++: Fast and Accurate LiDAR Semantic Segmentation](http://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf)
