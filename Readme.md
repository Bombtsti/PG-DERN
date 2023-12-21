# Property-guided few-shot learning for molecular property prediction with dual-view encoder and relation graph learning network

## Abstract

Molecular property prediction is an important task in drug discovery. However, experimental data for many drug molecules are limited, especially for novel molecular structures or rare diseases which affect the accuracy of many deep learning methods that rely on large training datasets. To this end, we propose PG-DERN, a novel few-shot learning model for molecular property prediction. A dual-view encoder is introduced to learn a meaningful molecular representation by integrating information from node and subgraph. Next, a relation graph learning module is proposed to construct a relation graph based on the similarity between molecules, which improves the efficiency of information propagation and the accuracy of property prediction. In addition, we use a MAML-based meta-learning strategy to learn well-initialized meta-parameters. In order to guide the tuning of meta-parameters, a property-guided feature augmentation module is designed to transfer information from similar properties to the novel property to improve the comprehensiveness of the feature representation of molecules with novel property. A series of comparative experiments on multiple benchmark datasets demonstrate that the proposed PG-DERN outperforms state-of-the-art methods.

![fig1](C:\Users\ZLW\Desktop\PG-DERN2\png\fig1.png)

## Environment

You can create a virtual environment as follows:

```
python==3.7.16
torch==1.7
torch-scatter==2.0.7
torch-sparse==0.6.9
torch-geometric==1.7.0
rdkit==2022.9.1
learn2learn==0.1.7
numpy==1.21.5
scikit-learn==1.0.2
```

## Datasets

The benchmark datasets are downloaded from the repository [Data](https://drive.google.com/file/d/1K3c4iCFHEKUuDVSGBtBYr8EOegvIJulO/view) from [Wang et al. (2021)](https://arxiv.org/abs/2107.07994).

## Run

1、tune the hyper-parameters in parser.py

2、run the main.py use the command:

```
python main.py
```

## baselines

```
Meta-MGNN(2021) - https://github.com/zhichunguo/Meta-MGNN
PAR(2021) - https://github.com/tata1661/PAR-NeurIPS21
HSL-RG(2023) - 
MTA(2023) -
```

## References

```
@InProceedings{wang2021property,
  title={Property-Aware Relation Networks for Few-Shot Molecular Property Prediction},
  author={Wang, Yaqing and Abuduweili, Abulikemu and Yao, Quanming and Dou, Dejing},
  booktitle = {Advances in Neural Information Processing Systems},
  year={2021},
}
```