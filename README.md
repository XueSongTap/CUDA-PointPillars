# PointPillars Inference with TensorRT

This repository contains sources and model for [pointpillars](https://arxiv.org/abs/1812.05784) inference using TensorRT.

Overall inference has below phases:

- Voxelize points cloud into 10-channel features
- Run TensorRT engine to get detection feature
- Parse detection feature and apply NMS

## Prerequisites

### Prepare Model && Data

We provide a [Dockerfile](docker/Dockerfile) to ease environment setup. Please execute the following command to build the docker image after nvidia-docker installation:
```
cd docker && docker build . -t pointpillar
```
We can then run the docker with the following command: 
```
nvidia-docker run --rm -ti -v /home/$USER/:/home/$USER/ --net=host --rm pointpillar:latest
```
For model exporting, please run the following command to clone pcdet repo and install custom CUDA extensions:
```
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet && git checkout 846cf3e && python3 setup.py develop
```
Download [PTM](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view) to ckpts/, then use below command to export ONNX model:
```
python3 tool/export_onnx.py --ckpt ckpts/pointpillar_7728.pth --out_dir model
```
Use below command to evaluate on kitti dataset, follow [Evaluation on Kitti](tool/eval/README.md) to get more detail for dataset preparation.
```
sh tool/evaluate_kitti_val.sh
```

### Setup Runtime Environment

- Nvidia Jetson Orin + CUDA 11.4 + cuDNN 8.9.0 + TensorRT 8.6.11

## Compile && Run

```shell
sudo apt-get install git-lfs && git lfs install
git clone https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars.git
cd CUDA-PointPillars && . tool/environment.sh
mkdir build && cd build
cmake .. && make -j$(nproc)
cd ../ && sh tool/build_trt_engine.sh
cd build && ./pointpillar
```

## FP16 Performance && Metrics

Average perf in FP16 on the training set(7481 instances) of KITTI dataset.

```
| Function(unit:ms) | Orin   |
| ----------------- | ------ |
| Voxelization      | 0.12   |
| Backbone & Head   | 6.09   |
| Decoder & NMS     | 1.58   |
| Overall           | 7.79   |
```

3D moderate metrics on the validation set(3769 instances) of KITTI dataset.

```
|                   | Car@R11 | Pedestrian@R11 | Cyclist@R11  | 
| ----------------- | --------| -------------- | ------------ |
| CUDA-PointPillars | 77.01   | 51.31          | 62.55        |
| OpenPCDet         | 77.28   | 52.29          | 62.68        |
```

## Note

- Voxelization has random output since GPU processes all points simultaneously while points selection for a voxel is random.

## References

- [Detecting Objects in Point Clouds with NVIDIA CUDA-Pointpillars](https://developer.nvidia.com/blog/detecting-objects-in-point-clouds-with-cuda-pointpillars/)
- [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)




## infer on 3080

```bash
root@yxc-MS-7B89:/home/yxc/code/for_new_cuda_pointpillar/CUDA-PointPillars/build# ./pointpillar ../data/  . --timer

GPU has cuda devices: 1
----device id: 0 info----
  GPU : NVIDIA GeForce RTX 3080 
  Capbility: 8.6
  Global memory: 10006MB
  Const memory: 64KB
  SM in a block: 48KB
  warp size: 32
  threads in a block: 1024
  block dim: (1024,1024,64)
  grid dim: (2147483647,65535,65535)

Total 10
------------------------------------------------------
Lidar Backbone 🌱 is Static Shape model
Inputs: 3
        0.voxels : {10000 x 32 x 10} [Float16]
        1.voxel_idxs : {10000 x 4} [Int32]
        2.voxel_num : {1} [Int32]
Outputs: 3
        0.cls_preds : {1 x 248 x 216 x 18} [Float32]
        1.box_preds : {1 x 248 x 216 x 42} [Float32]
        2.dir_cls_preds : {1 x 248 x 216 x 12} [Float32]
------------------------------------------------------

<<<<<<<<<<<
Load file: ../data/000003.bin
Lidar points count: 18911
==================PointPillars===================
[⏰ [NoSt] CopyLidar]:  0.12493 ms
[⏰ Lidar Voxelization]:        0.07690 ms
[⏰ Lidar Backbone & Head]:     2.70435 ms
[⏰ Lidar Decoder + NMS]:       3.06573 ms
Total: 5.847 ms
=============================================
Detections after NMS: 5
Saved prediction in: .000003.txt
>>>>>>>>>>>

<<<<<<<<<<<
Load file: ../data/000000.bin
Lidar points count: 20285
==================PointPillars===================
[⏰ [NoSt] CopyLidar]:  0.07651 ms
[⏰ Lidar Voxelization]:        0.03757 ms
[⏰ Lidar Backbone & Head]:     1.69472 ms
[⏰ Lidar Decoder + NMS]:       2.92045 ms
Total: 4.653 ms
=============================================
Detections after NMS: 8
Saved prediction in: .000000.txt
>>>>>>>>>>>

<<<<<<<<<<<
Load file: ../data/000006.bin
Lidar points count: 19473
==================PointPillars===================
[⏰ [NoSt] CopyLidar]:  0.07597 ms
[⏰ Lidar Voxelization]:        0.04058 ms
[⏰ Lidar Backbone & Head]:     1.70496 ms
[⏰ Lidar Decoder + NMS]:       3.04742 ms
Total: 4.793 ms
=============================================
Detections after NMS: 18
Saved prediction in: .000006.txt
>>>>>>>>>>>

<<<<<<<<<<<
Load file: ../data/000008.bin
Lidar points count: 17238
==================PointPillars===================
[⏰ [NoSt] CopyLidar]:  0.06371 ms
[⏰ Lidar Voxelization]:        0.03725 ms
[⏰ Lidar Backbone & Head]:     1.69165 ms
[⏰ Lidar Decoder + NMS]:       2.99734 ms
Total: 4.726 ms
=============================================
Detections after NMS: 24
Saved prediction in: .000008.txt
>>>>>>>>>>>

<<<<<<<<<<<
Load file: ../data/000005.bin
Lidar points count: 19962
==================PointPillars===================
[⏰ [NoSt] CopyLidar]:  0.07789 ms
[⏰ Lidar Voxelization]:        0.04506 ms
[⏰ Lidar Backbone & Head]:     1.70496 ms
[⏰ Lidar Decoder + NMS]:       3.01453 ms
Total: 4.765 ms
=============================================
Detections after NMS: 8
Saved prediction in: .000005.txt
>>>>>>>>>>>

<<<<<<<<<<<
Load file: ../data/000004.bin
Lidar points count: 19063
==================PointPillars===================
[⏰ [NoSt] CopyLidar]:  0.06710 ms
[⏰ Lidar Voxelization]:        0.04054 ms
[⏰ Lidar Backbone & Head]:     1.70189 ms
[⏰ Lidar Decoder + NMS]:       3.00029 ms
Total: 4.743 ms
=============================================
Detections after NMS: 15
Saved prediction in: .000004.txt
>>>>>>>>>>>

<<<<<<<<<<<
Load file: ../data/000002.bin
Lidar points count: 20210
==================PointPillars===================
[⏰ [NoSt] CopyLidar]:  0.07466 ms
[⏰ Lidar Voxelization]:        0.03731 ms
[⏰ Lidar Backbone & Head]:     1.69472 ms
[⏰ Lidar Decoder + NMS]:       2.95325 ms
Total: 4.685 ms
=============================================
Detections after NMS: 14
Saved prediction in: .000002.txt
>>>>>>>>>>>

<<<<<<<<<<<
Load file: ../data/000007.bin
Lidar points count: 19423
==================PointPillars===================
[⏰ [NoSt] CopyLidar]:  0.07462 ms
[⏰ Lidar Voxelization]:        0.04138 ms
[⏰ Lidar Backbone & Head]:     1.70496 ms
[⏰ Lidar Decoder + NMS]:       3.06074 ms
Total: 4.807 ms
=============================================
Detections after NMS: 12
Saved prediction in: .000007.txt
>>>>>>>>>>>

<<<<<<<<<<<
Load file: ../data/000009.bin
Lidar points count: 19411
==================PointPillars===================
[⏰ [NoSt] CopyLidar]:  0.07024 ms
[⏰ Lidar Voxelization]:        0.04096 ms
[⏰ Lidar Backbone & Head]:     1.70291 ms
[⏰ Lidar Decoder + NMS]:       3.11091 ms
Total: 4.855 ms
=============================================
Detections after NMS: 13
Saved prediction in: .000009.txt
>>>>>>>>>>>

<<<<<<<<<<<
Load file: ../data/000001.bin
Lidar points count: 18630
==================PointPillars===================
[⏰ [NoSt] CopyLidar]:  0.07107 ms
[⏰ Lidar Voxelization]:        0.04045 ms
[⏰ Lidar Backbone & Head]:     1.70086 ms
[⏰ Lidar Decoder + NMS]:       3.13741 ms
Total: 4.879 ms
=============================================
Detections after NMS: 10
Saved prediction in: .000001.txt
>>>>>>>>>>>
```