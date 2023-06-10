# ESM

This is our Pytorch implementation for the paper:

Pengyu Zhao; Shoujin Wang; Wenpeng Lu*; Xueping Peng;  Weiyu Zhang; Chaoqun Zheng; Yonggang Huang. News Recommendation via Jointly Modeling Event Matching and Style Matching, European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases 2023.


The models trained by us can be downloaded from Baidu Netdisk:   

# Preparation
## 0. Requirement
python=3.6.13
pytorch=1.10.0
cudatoolkit = 10.2.89
numpy = 1.19.5
scikit-learn=0.24.2
torchtext = 0.11.0
torchvision=0.11.0

## 1.Data Preparation
The dataset is MIND dataset.

MIND: A Large-scale Dataset for News Recommendation, https://aclanthology.org/2020.acl-main.331/.

## 2.Data Preprocess
python ESM_500K/prepare_MIND_dataset.py

python ESM_Large/prepare_MIND_dataset.py

## 3.Train
python ESM_500K/main.py    
 
python ESM_Large/main.py   

## 4.Generate Event Channel Lable
We provide the file of generated event channel label. If you want to obtain the generated event channel label by yourself, you need to run the followed code:

python ESM_Large/tfidf_17.py

python ESM_Large/tfidf_10.py






