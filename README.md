# Bird Classification by efficientNet

## Objective
Classify 200 bird species. 

## Requirement

1. conda create environment from file.
```bash
conda env create -f environment.yml
conda activate TransFG
```
> May show some error. Just ignore them.

2. install pytorch correspond to your cuda version.

python 3.7

PyTorch >= 1.5.1

torchvision >= 0.6.1


## Data preparation

1. Download datatsets. [link](https://drive.google.com/drive/folders/11nUVfbylNeJ3zl3AUbCkn9_8c4Os1GbG?usp=sharing)

> 以上連結，資料格式已經處理好了 

2. organize the structure as follows:
```
data
├── classes.txt
├── training_labels.txt
├── testing_img_order.txt
├── train
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── test
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

* Data information:
    * Consist 200 classes of birds, and each category contains 15 pictures.
    * (Training: 3000, Testing: 3033)
    
> "training_labels.txt" specify the label of training data. 
> "testing_img_order.txt" is the test order. (No Label)
    
## Execute

1. Train: "python train.py"
2. Test: "python test.py"

> PS. Modify config/config.yml for further adjustment.
    
## Pretrained model
* efficientnetb2: [effb2_R4_ep7_vloss1.844_LR9.871247457565971e-06.pth](https://drive.google.com/drive/u/1/folders/11nUVfbylNeJ3zl3AUbCkn9_8c4Os1GbG)

* Train from scratch: download pretrained weight ["efficientnet-b2-8bb594d6.pth"](https://github.com/lukemelas/EfficientNet-PyTorch/releases)

## 作法

* 做法：EfficientNet-b1
* Learning rate scheduler: warmup_cosine, warmup_linear, ReduceLROnPlateau
* Optimizer: Adam
* Loss function: cross entropy with label smoothing mechanism.
* Random augmentation (N, M) = (2, 10)


## 資料

鳥類資料：
原圖解析度：(300 ~ 500) * (300 ~ 500)
200種鳥類圖片

* train valid 資料分割：
隨機分配8:2


* 訓練資料前處理：
1. Resize成解析度[224, 224]
2. 隨機水平、垂直翻轉
3. 隨機旋轉三十度
4. 隨機位移20%的圖片長
5. Normalize as ImageNet

* 測試資料前處理：
1. Resize成解析度[224, 224]

## 結果
smooth_CrossEntropy: 

Valid Loss: 1.840

Valid Acc: 70.0 %

Test score: 0.5536





