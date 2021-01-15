# Convert .npy to .ply

## Usage

If you want to convert .npy to .ply please run the command below

```bash 

python npy2ply.py \
        --rawdata_path [default == "./input.npy"] \
        --gt_path [default == "./seg_gt.npy"] \
        --pred_path [default == "./seg_pred.npy"] \

rawdata_path: The original data. data shape is (n,m,k), n is num of blocks, m is point information(XYZRGB...), k is num of points

gt_path: The ground truth label of every points. data shape is (n), n is the num of points.

pred_pred: The predict result of every points. data shape is (n), n is the num of points.

```
The labels corresponding to each color are as follows:

![image](https://github.com/KaivinC/Pointcloud-npy2ply/blob/master/image/colors.png)
