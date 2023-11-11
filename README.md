# Semantic Segmentation of LiDAR Point Cloud for Autonomy.
Semantic Segmentation for full 3D Lidar point cloud in real time.
Master of Science in Robotics Engineering at [Worcester Polytechnic Institute](https://www.wpi.edu/)

## Project Description

### Dependencies

- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [NVIDIA CuDNN](https://developer.nvidia.com/cudnn)
- [Tensorflow](https://www.tensorflow.org/install)
- NumPy
- Matplotlib
- Pillow

### Dataset - Semantic KITTI

Download the Velodyne Sensor data and Label data, and place in the `dataset` folder in the form as mentioned on the [Semantic KITTI website](http://www.semantic-kitti.org/dataset.html#overview).

- [Download](http://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip) Point Cloud Data
- Download Labels Data -> Save this link -> http://www.semantic-kitti.org/assets/data_odometry_labels.zip

The path of this `dataset` folder will be required as a argument to the run the entire pipeline.

## Usage Guidelines:

Go to the parent folder of this repo, that is, [semantic_segmentation](.) and enter the command:
```
python3 scripts/main.py -d **path_to_dataset_folder**
```

## References

1. A. Milioto and I. Vizzo and J. Behley and C. Stachniss, [RangeNet++: Fast and Accurate LiDAR Semantic Segmentation](http://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf)
