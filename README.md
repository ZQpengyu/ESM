# ESM

This is our Pytorch implementation for the paper:

Pengyu Zhao; Shoujin Wang; Wenpeng Lu*; Xueping Peng;  Weiyu Zhang; Chaoqun Zheng; Yonggang Huang. News Recommendation via Jointly Modeling Event Matching and Style Matching, European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases 2023.


The models trained by us and the splitted data can be downloaded from Baidu Netdisk:  https://pan.baidu.com/s/1f7iZNaebsw02X0zXvswb4A   password:jncb

# Preparation
## Requirement
python=3.6.13

pytorch=1.10.0

cudatoolkit = 10.2

numpy = 1.19.5

scikit-learn=0.24.2

torchtext = 0.11.0

torchvision=0.11.0


## Data Preparation
The dataset is MIND dataset.

MIND: A Large-scale Dataset for News Recommendation, https://aclanthology.org/2020.acl-main.331/.

## Data Preprocess
We have uploaded the processed dataset to Baidu Netdisk. If you wish to handle it yourself, you can run the following files separately.

python ESM_500K/prepare_MIND_dataset.py

python ESM_Large/prepare_MIND_dataset.py

## Train
python ESM_500K/main.py    
 
python ESM_Large/main.py   

## Generate Event Channel Lable
We have already provided the generated event channel labels, which are stored in the corresponding folder. If you want to generate the labels yourself, please run the following code

python ESM_Large/tfidf_17.py

python ESM_Large/tfidf_10.py



## Acknowledgement
Any scientific publications that use our codes should cite the following paper as the reference:
<pre><code>
 @inproceedings{ESM-2023, 
    author    = {Pengyu Zhao,
                Shoujin Wang,
                Wenpeng Lu, 
                Xueping Peng,  
                Weiyu Zhang,
                Chaoqun Zheng, 
                Yonggang Huang }, 
   title     = {News Recommendation via Jointly Modeling Event Matching and Style Matching}, 
   booktitle   = {{Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases}}, 
   year      = {2023} 
 }
</code></pre>






