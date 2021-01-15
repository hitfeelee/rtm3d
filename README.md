# rtm3d
Perform 3D object detection base on anchor-free. rtm3d on main branch, smoke on smoke branch.
There are many changes both rtm3d and smoke, and main keypoint heatmap in smoke base on rtm3d. 
Train smoke also base on rtm3d.


## Quick Start
### datasets
Applying kitt dataset. Place [kitti_dev](https://github.com/hitfeelee/kitti_dev) sub contents to datasets/data/kitti/ of this project.

Please place it as following:

    root
    |
    ---datasets
       |
       ---data
          |
          ---kitti
              ---cache
                 |
                 ---k_*.npy // list K of camera. * -> (train or test)
                 |
                 ---label_*.npy // list label. * -> (train or test) 
                 |
                 ---shape_*.npy // list size of images. * -> (train or test)
              |
              ---ImageSets
                 |
                 ---train.txt // list of training image.
                 |
                 ---test.txt // list of testing image.
              ---testing
              |
              ---training
                 |
                 ---calib
                    |
                    ---calib_cam_to_cam.txt // camera calibration file for kitti
                 |
                 ---image_2
                 |
                 ---label_2
          

### training

#### single gpu
    python train.py --model-config ./models/configs/rtm3d_dla34_kitti.yaml 

#### multi gpu
    python train_multi_gpu.py --dist-url 'tcp://127.0.0.1:23456'  --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --model-config ./models/configs/rtm3d_resnet18_kitti.yaml 


### detect
    python detect.py --model-config ./models/configs/rtm3d_resnet18_kitti.yaml
    
### export onnx
     python export_onnx.py --model-config ./models/configs/rtm3d_resnet18_kitti.yaml
    
## Pretrained Model
We provide a set of trained models available for download in the  [Pretrained Model](https://pan.baidu.com/s/1G7pI7Gl-UROfNzyMxnWGtg).
提取码: dxv4

## License
MIT

## Refenrence
smoke.
rtm3d.