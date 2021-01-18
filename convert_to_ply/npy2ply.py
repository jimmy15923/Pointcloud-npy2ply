import os
import time
import numpy as np
import ply
import random
import cv2
import copy
import argparse


def get_args():
    parser = argparse.ArgumentParser("S3DIS")

    parser.add_argument("--data_path", type = str, default = "./input.npy")
    parser.add_argument("--gt_path", type = str, default = "./seg_gt.npy")
    parser.add_argument("--pred_path", type = str, default = "./seg_pred.npy")
    args = parser.parse_args()

    return args

def npy2ply(all_data,name):
    for i in range(all_data.shape[0]):
        ply_path = ".\exp"
        if not os.path.exists(ply_path):
            os.makedirs(ply_path)

        cloud_file = os.path.join(ply_path, name + str(i) + '.ply')

        cloud_points = np.empty((0, 3), dtype=np.float32)
        cloud_colors = np.empty((0, 3), dtype=np.uint8)
        cloud_classes = np.empty((0, 1), dtype=np.int32)

        object_data = all_data[i]
        object_class = all_data[i][:,6].astype(np.uint8)
        
        cloud_points = np.vstack((cloud_points, object_data[:, 0:3].astype(np.float32)))
        cloud_colors = np.vstack((cloud_colors, colors[object_class].astype(np.uint8)))
        cloud_classes = np.vstack((cloud_classes, object_class.reshape(object_class.shape[0],1)))

        print(cloud_file)
        ply.write_ply(cloud_file,
              (cloud_points, cloud_colors, cloud_classes),
              ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

args = get_args()

#S3DIS lebel
label_to_names = {0: 'ceiling',
                       1: 'floor',
                       2: 'wall',
                       3: 'beam',
                       4: 'column',
                       5: 'window',
                       6: 'door',
                       7: 'chair',
                       8: 'table',
                       9: 'bookcase',
                       10: 'sofa',
                       11: 'board',
                       12: 'clutter'}
name_to_label = {v: k for k, v in label_to_names.items()}


# generate random color
colors = [
        [233, 229, 107],  # 'ceiling' .-> .yellow
        [95, 156, 196],  # 'floor' .-> . blue
        [179, 116, 81],  # 'wall'  ->  brown
        [241, 149, 131],  # 'beam'  ->  salmon
        [81, 163, 148],  # 'column'  ->  bluegreen
        [77, 174, 84],  # 'window'  ->  bright green
        [108, 135, 75],  # 'door'   ->  dark green
        [41, 49, 101],  # 'chair'  ->  darkblue
        [79, 79, 76],  # 'table'  ->  dark grey
        [223, 52, 52],  # 'bookcase'  ->  red
        [89, 47, 95],  # 'sofa'  ->  purple
        [81, 109, 114],  # 'board'   ->  grey
        [233, 233, 229],  # 'clutter'  ->  light grey
        [0, 0, 0]  # unlabelled .->. black
]
colors = np.array(colors)


# #generate color lengend
# if(True):
#     color_path = "./color_legend"
#     if not os.path.exists(color_path):
#         os.makedirs(color_path)

#     color_legend = copy.deepcopy(colors)
#     color_legend[:,[2,0]] = color_legend[:,[0,2]]
#     for i in range(13):
#         image = np.repeat(np.repeat(color_legend[i].reshape(1,3),128,axis=0).reshape(1,128,3),128,axis=0)
#         cv2.imwrite(os.path.join(color_path,label_to_names[i]+'_.png'), image)


raw_data = np.load(args.data_path)
raw_data = raw_data.swapaxes(1,2)
raw_data = raw_data[:,:,0:6]

gt_data = np.load(args.gt_path)
predict_data = np.load(args.pred_path)

all_gt_data = np.concatenate((raw_data, gt_data.reshape((gt_data.shape[0],gt_data.shape[1],1))) ,axis=2)
all_predict_data = np.concatenate((raw_data, predict_data.reshape((predict_data.shape[0],predict_data.shape[1],1))),axis=2)



npy2ply(all_gt_data,"gt")
npy2ply(all_predict_data,"predict")
