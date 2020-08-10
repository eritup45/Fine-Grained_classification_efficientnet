# Spinneret Classification

### Execute:
    1. Download data, and put in "machine_data". 
    > Remember to classify data into "ng" and "ok" folders. Split data into "Train"(75%) "Test"(25%) folders.
    
    2. Train: "python train.py"
    3. Test: "python test.py"
    
    > PS. Modify config/config.yml for further adjustment.
    
### 作法

做法：先將small歸類於ng，再訓練，

模型：efficientnet lite0(pretrained)，最後一層改成
```python=
self.prob_model = nn.Sequential(
            nn.Linear(eff_lite.classifier.in_features, 128),
            nn.Linear(128, 8),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
```

### 資料

十字孔資料：
原圖解析度：1056*1056
ng張數：3565
ok張數：79
small張數：460

訓練資料前處理：
1. Resize成解析度[528, 528]
2. 隨機水平、垂直翻轉
3. 隨機旋轉九十度
4. 隨機位移20%的圖片長

測試資料前處理：
1. Resize成解析度[528, 528]

### 結果
Train Loss：

Test：
1. TEST_THRESHOLD = 0.5
![](https://i.imgur.com/wxveyn8.png)

2. TEST_THRESHOLD = 0.8
![](https://i.imgur.com/KISAFH7.png)

> confusion matrix:
> ans\pred: ng, ok
> ng [ 831, 60 ]
> ok [ 4,   15 ]
> 

---------
### 失敗做法與模型：

#### 將small不歸類於任何一類：
從頭訓練：
==(成功)==
![](https://i.imgur.com/tWLtidE.png)


![](https://i.imgur.com/vNUIpra.png)

weight: eff_lite/**epoch60_loss0.135_acc95_model.pth**

model 資訊:
![](https://i.imgur.com/4KBfM9L.png)

---
==(成功)==
![](https://i.imgur.com/RjhYIuQ.png)

weight: **test_loss0.19_acc89_model.pth**

---

(收斂失敗)
![](https://i.imgur.com/2p2cap2.png)

![](https://i.imgur.com/XEvo4ll.png)


(做法有錯) (load 之前訓練的small歸類到ng)
![](https://i.imgur.com/bC8RXSE.png)

![](https://i.imgur.com/Dtjobhz.png)

使用efficient lite0 pretrained model，freeze最後五層
![](https://i.imgur.com/PM77nqX.png)

![](https://i.imgur.com/t938nz9.png)


#### 將small歸類於ng：

從頭訓練:
![](https://i.imgur.com/WiTimT5.png)

![](https://i.imgur.com/ETyEsuL.png)

---

![](https://i.imgur.com/V0Aawnh.png)

![](https://i.imgur.com/lYsRKEd.png)

---

![](https://i.imgur.com/yDjfs3K.png)

![](https://i.imgur.com/506OW4d.png)









