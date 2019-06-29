# LSTA: Long Short-Term Attention for Egocentric Action Recognition


We release the PyTorch code of [LSTA](https://arxiv.org/pdf/1811.10698.pdf)

![LSTA](https://drive.google.com/uc?export=view&id=1gf9Ih_mK1xsd4ZVZvP7tsy4QJEkK1Dsz)


#### Reference
Please cite our paper if you find the repo and the paper useful.
```
@InProceedings{Sudhakaran_2019_CVPR,
author = {Sudhakaran, Swathikiran and Escalera, Sergio and Lanz, Oswald},
title = {{LSTA: Long Short-Term Attention for Egocentric Action Recognition}},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

#### Prerequisites

* Python 3.5
* Pytorch 0.3.1


#### Training

* ##### RGB
To train the models, run the script train_rgb.sh, which contains:
````
python main_rgb.py --dataset gtea_61 --root_dir dataset --outDir experiments --stage 1 \
                   --seqLen 25 --trainBatchSize 32 --numEpochs 200 --lr 0.001 --stepSize 25 75 150 \
                   --decayRate 0.1 --memSize 512 --outPoolSize 100 --evalInterval 5 --split 2
````

#### Evaluation
Testing on the trained models can be done by running the script test_rgb.sh
#### **Pretrained models**

The pre-trained models can be downloaded from the following [Google Drive link](https://drive.google.com/drive/folders/1KIUuoaa1_ipGFOYZB6Oe3yITBKZlrpWr?usp=sharing)



#### TODO
1. EPIC-KITCHENS code
2. Flow and two stream codes
3. Pre-trained models



