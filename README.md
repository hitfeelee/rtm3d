# rtm3d
Perform Real-time 3D object detection base on anchor-free. rtm3d on main branch , smoke on smoke branch and simple-smoke. This branch inherited from smoke branch, but simplify it as many as possible.
There are many changes both rtm3d and smoke, and main keypoint heatmap in smoke base on rtm3d. 
Train smoke also base on rtm3d. And now multi gpu mix-float training in this branch has supported.


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
              |
              ---tracking
                 |
                 ---training
                     |
                     ---calib
                     |
                     ---image_2
                 |
                 ---testing
                     |
                     ---calib
                     |
                     ---image_2

          

### training


You can specific gpu num in train_multi_gpu.py, like as os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    ./train.sh 

### detect
    python detect.py --model-config ./models/configs/rtm3d_resnet_kitti.yaml

### detect_seq
Specific dataset root path to "./datasets/data/kitti/tracking" by configure param DATASET.PATH in rtm3d_resnet_kitti.yaml
    
    python detect_seq.py --model-config ./models/configs/rtm3d_resnet_kitti.yaml
    
### export onnx
     python export_onnx.py --model-config ./models/configs/rtm3d_resnet_kitti.yaml
    
## Pretrained Model
We provide a set of trained models available for download in the  [Pretrained Model](https://pan.baidu.com/s/15ZC5IglSJtVXgEvJc8vOfg). Password: lbkj

car iou: 0.7/0.5/0.5

pedestrian iou: 0.5/0.4/0.4

cyclist iou: 0.5/0.4/0.4

| class | mAP (easy/moderate/hard) |
| :------| :------ | 
| car | 39.15/33.59/29.81 |
| pedestrian | 12.22/9.58/9.32 |
| cyclist | 16.24/13.44/13.31 |


## License
MIT

## Refenrence
smoke.

rtm3d.