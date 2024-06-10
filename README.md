## Generalized Unbiased Scene Graph Generation

![LICENSE](https://img.shields.io/badge/license-MIT-green)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.2.0-%237732a8)

This repository contains code for the paper "[Generalized Unbiased Scene Graph Generation](https://arxiv.org/pdf/2308.04802)". This code is based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch). 

## Abstract

Existing Unbiased Scene Graph Generation (USGG) methods only focus on addressing the predicate-level imbalance that high-frequency classes dominate predictions of rare ones, while overlooking the concept-level imbalance.
Actually, even if predicates themselves are balanced, there is still a significant concept-imbalance within them due to the long-tailed distribution of contexts (i.e., subject-object combinations). 
This concept-level imbalance poses a more pervasive and challenging issue compared to the predicate-level imbalance since subject-object pairs are inherently complex in combinations.
To address the issue, we propose Multi-Concept Learning (MCL), a novel concept-level balanced learning framework orthogonal to existing SGG methods.
MCL first quantifies the concept-level imbalance across predicates in terms of different amounts of concepts, representing as multiple concept-prototypes within the same class.
It then effectively learns concept-prototypes by applying the Concept Regularization (CR) technique. 
Furthermore, to achieve balanced learning over different concepts, we introduce the Concept-based Balanced Memory (CBM), which guides SGG models to generate balanced representations for concept-prototypes.
Finally, we introduce a novel metric, mean Context Recall (mCR@K), as a complement to mean Recall (mR@K), to evaluate the model's performance across concepts (determined by contexts) within the same predicate. 
Extensive experiments demonstrate the remarkable efficacy of our model-agnostic strategy in enhancing the performance of benchmark models on both VG-SGG and OI-SGG datasets, leading to new state-of-the-art achievements in two key aspects: predicate-level unbiased relation recognition and concept-level compositional generability. 
<div align=center><img height="600" width="600"  src=abstract.png></div>

## Framework
WThe Overview of our Multi-Concept Learning (MCL) framework. It contains three parts: Concept-Prototype Construction (CPC), Concept-based Balanced Memory (CBM), and Concept Regularization (CR). The CPC assesses the semantic scales of each predicate from the SGG dataset and then quantifies the semantic scales into different amounts of concept-prototypes within the same predicate. Subsequently, CBM produces the balanced relation representations across different concepts within predicates, enabling the SGG model to equally attend to all concepts. Finally, the CR alleviates the predicate-level semantic overlap while enhancing the discriminability among concept-prototypes.
<div align=center><img src=framework.png></div>

## Visualization
<div align=center><img  height="600" width="800" src=visual_sp-1.png></div>


## Dataset
Follow [DATASET.md](DATASET.md) for instructions of dataset preprocessing.

## Train
Follow the [instructions](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) to install and use the code. Also, we provide scripts for training models with MCL our model (in `scripts/885train_trans.sh`(https://github.com/XinyuLyu/G-USGG/blob/master/885train_trans.sh))
    
## Test
The trained models(Motif-MCL) on Predcls\SGCLs\SGDet are released as below. 


| Predcls                                                                                                                                                           | SGCLs                                                                                                                                                            | SGDet                                                                                                                                                           |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Motif-MCL-Predcls](https://1drv.ms/f/s!Amlnn8hF2cFdgul3oSr1pHfa7eJCKA?e=1czlAO) | [Motif-FMCL-SGCLS](https://1drv.ms/f/s!Amlnn8hF2cFdgul5g5ZxuviLtp5YMA?e=JYhsTy) | [Motif-MCL-SGDet](https://1drv.ms/f/s!Amlnn8hF2cFdgul4LohJW7XzV_1SEw?e=YZKlXA) |

## Help
Be free to contact me (xinyulyu68@gmail.com) if you have any questions!

## Acknowledgement
The code is implemented based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), and [SGG-G2S](https://github.com/ZhuGeKongKong/SGG-G2S). Thanks for their great works! 

## Bibtex

```
@inproceedings{sgg:MCL,
  author    = {Xinyu Lyu and
               Lianli Gao and
               Junlin Xie and
               Pengpeng Zeng and
               Yulu Tian and
               Jie Shao and 
               Heng Tao Shen},
  title     = {Generalized Unbiased Scene Graph Generation},
  booktitle = {CoRR},
  year      = {2023}
}
```
