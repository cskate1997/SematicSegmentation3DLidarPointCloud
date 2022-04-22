# import pcl
import numpy as np
import yaml
# import pcl.pcl_visualization

def read_yaml_color(path):
    with open(path, "r") as stream:
        try:
            yamlFile = yaml.safe_load(stream)
            # print(yamlFile)
        except yaml.YAMLError as exc:
            print(exc)
    colors = yamlFile['color_map']
    return colors

def read_bin_as_array(path):
    bin_pcd = np.fromfile(path, dtype=np.float32)
    points = bin_pcd.reshape((-1, 4))[:, 0:4]
    return points

def read_labels(path):
    # print("path = ", type(path))
    label_pcd = np.fromfile(path, dtype=np.uint32)
    # labels = {}
    labelList = []
    for i in range(label_pcd.shape[0]):
        label = label_pcd[i] & 0x0000FFFF
        labelList.append(label)
        # instance = ((label_pcd[i] & 0xFFFF0000) >> 16)
        # if label not in labels:

        #     labels[label] = {}
        #     if instance not in labels[label]:
        #         labels[label][instance] = 1
        #     else:
        #         labels[label][instance] += 1
        # else:
        #     # labels[label] += 1
        #     if instance not in labels[label]:
        #         labels[label][instance] = 1
        #     else:
        #         labels[label][instance] += 1
    # return labels
    return labelList

# def read_labels(path, points, colors):
    # label_pcd = np.fromfile(path, dtype=np.uint32)
    # labels = {}
    # l = []
    # for i in range(label_pcd.shape[0]):
    #     label = label_pcd[i] & 0x0000FFFF
    #     instance = ((label_pcd[i] & 0xFFFF0000) >> 16)
    #     color = ((colors[label][0] & 0xFF) << 16) | ((colors[label][1] & 0xFF) << 8) | (colors[label][2] & 0xFF)
    #     l.append([points[i,0], points[i,1], points[i,2], color ])
    #     if label not in labels:

    #         labels[label] = {}
    #         if instance not in labels[label]:
    #             labels[label][instance] = 1
    #         else:
    #             labels[label][instance] += 1
    #     else:
    #         # labels[label] += 1
    #         if instance not in labels[label]:
    #             labels[label][instance] = 1
    #         else:
    #             labels[label][instance] += 1
    # return labels, l

def save_point_cloud_as_pcd(path, cloud):
    cloud.to_file(bytes(path, 'utf-8'))

def convert_to_point_cloud(points, format="XYZ"):
    if format == "XYZ":
        cloud = pcl.PointCloud()
    elif format == "XYZI":    
        cloud = pcl.PointCloud_PointXYZI()
    elif format == "XYZRGB":
        cloud = pcl.PointCloud_PointXYZRGB()
    else:
        print("unknown format requested: " + format)
        return None
    if isinstance(points, list):
        try:
            cloud.from_list(points)
        except Exception as e:
            print(e)
            return None
    elif isinstance(points, np.ndarray):
        try:
            cloud.from_array(points)
        except Exception as e:
            print(e)
            return None
    return cloud

def visualize_point_cloud(cloud, format="XYZ", block=True):
    visual = pcl.pcl_visualization.CloudViewing()
    if format == "XYZ":
        visual.ShowMonochromeCloud(cloud)
    elif format == "XYZI":    
        visual.ShowGrayCloud(cloud)
    elif format == "XYZRGB":
        visual.ShowColorCloud(cloud)
    else:
        print("unknown format requested: " + format)
        return None

    while block:
        block = not(visual.WasStopped())
    
    return visual


if __name__ == '__main__':
    colors = read_yaml_color("data/semantic-kitti.yaml")
    points = read_bin_as_array("data/sequences/00/velodyne/000000.bin")
    labels, l = read_labels("data/sequences/00/labels/000000.label", points, colors)
    cloud = convert_to_point_cloud(l, "XYZRGB")
    visualize_point_cloud(cloud, "XYZRGB")