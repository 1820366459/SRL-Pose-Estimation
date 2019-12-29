# SRL for Human Pose Estimation

## Introduction
This is an official pytorch implementation of **Fast Non-Local Neural Networks with Spectral Residual Learning**.

This repo uses [*Simple Baselines*](http://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html) as the baseline method for Pose Estimation. 

## Main Results
### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset
| Arch | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|---|---|---|---|---|---|---|---|---|---|
| 256x192_pose_resnet_50_SRL_FFT | 0.709 | 0.891 | 0.785 | 0.674 | 0.779 | 0.768 | 0.932 | 0.836 | 0.723 | 0.832 |
| 384x288_pose_resnet_50_SRL_FFT | 0.733 | 0.895 | 0.800 | 0.694 | 0.806 | 0.786 | 0.934 | 0.848 | 0.740 | 0.853 |
| 512x384_pose_resnet_50_SRL_FFT | 0.738 | 0.897 | 0.803 | 0.696 | 0.815 | 0.790 | 0.934 | 0.849 | 0.741 | 0.860 |
| 256x192_pose_resnet_101_SRL_FFT | 0.718 | 0.893 | 0.796 | 0.684 | 0.787 | 0.776 | 0.935 | 0.844 | 0.733 | 0.839 |
| 384x288_pose_resnet_101_SRL_FFT | 0.743 | 0.901 | 0.813 | 0.705 | 0.815 | 0.797 | 0.941 | 0.860 | 0.753 | 0.861 |
| 512x384_pose_resnet_101_SRL_FFT | 0.749 | 0.899 | 0.816 | 0.711 | 0.823 | 0.801 | 0.938 | 0.862 | 0.757 | 0.865 |
| 256x192_pose_resnet_152_SRL_FFT | 0.721 | 0.895 | 0.797 | 0.688 | 0.791 | 0.780 | 0.937 | 0.849 | 0.737 | 0.842 |
| 384x288_pose_resnet_152_SRL_FFT | 0.746 | 0.897 | 0.817 | 0.708 | 0.819 | 0.801 | 0.939 | 0.864 | 0.757 | 0.864 |
| 512x384_pose_resnet_152_SRL_FFT | 0.753 | 0.902 | 0.817 | 0.715 | 0.826 | 0.804 | 0.940 | 0.861 | 0.760 | 0.868 |


### Note:
- Flip test is used.
- Person detector has person AP of 56.4 on COCO val2017 dataset.

## Environment
The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using 8 NVIDIA P100 GPU cards. Other platforms or GPU cards are not fully tested.

## Quick start
### Installation
1. Install pytorch >= v1.0.0 following [official instruction](https://pytorch.org/).
   **Note that if you use pytorch's version < v1.0.0, you should following the instruction at <https://github.com/Microsoft/human-pose-estimation.pytorch> to disable cudnn's implementations of BatchNorm layer. We encourage you to use higher pytorch's version(>=v1.0.0)**
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
5. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
6. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── tools 
   ├── README.md
   └── requirements.txt
   ```
7. Download pytorch imagenet pretrained models from [pytorch model zoo](https://pytorch.org/docs/stable/model_zoo.html#module-torch.utils.model_zoo). Please download them under ${POSE_ROOT}/models/pytorch, and make them look like this:

   ```
   ${POSE_ROOT}
    `-- models
        `-- pytorch
            `-- imagenet
                |-- resnet50-19c8e357.pth
                |-- resnet101-5d3b4d8f.pth
                |-- resnet152-b121ed2d.pth
   ```

### Data preparation
Please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. We also provide person detection result of COCO val2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

### Training on COCO train2017

```
python pose_estimation/train.py \
    --cfg experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml \
    --dec_nl 1 1 1 --nltype fft
```

### Valid on COCO val2017

```
python pose_estimation/valid.py \
    --cfg experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml \
    --flip-test \
    --use-detect-bbox \
    --dec_nl 1 1 1 --nltype fft
```

### Citation
If you use our code or models in your research, please cite with:
```
@inproceedings{chi2019SRL,
    author={Chi, Lu and Tian, Guiyu and Mu, Yadong and Xie, Lingxi and Tian, Qi},
    title={Fast Non-Local Neural Networks with Spectral Residual Learning},
    booktitle = {ACM International Conference on Multimedia (MM)},
    year = {2019}
}
```
# Thanks to the Third Party Libs
- [Simple Baselines for Human Pose Estimation and Tracking](https://github.com/microsoft/human-pose-estimation.pytorch)
- [Compact Generalized Non-local Network](https://github.com/KaiyuYue/cgnl-network.pytorch.git)
- [CCNet: Criss-Cross Attention for Semantic Segmentation](https://github.com/speedinghzl/CCNet.git)
