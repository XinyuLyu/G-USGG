## Generalized Unbiased Scene Graph Generation

![LICENSE](https://img.shields.io/badge/license-MIT-green)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.2.0-%237732a8)

This repository contains code for the paper "[Generalized Unbiased Scene Graph Generation](https://arxiv.org/pdf/2308.04802)". This code is based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch). 

## Abstract

Existing Unbiased Scene Graph Generation (USGG) methods only focus on addressing the predicate-level imbalance that high-frequency classes dominate predictions of rare ones, while overlooking the concept-level imbalance.
Actually, even if predicates themselves are balanced, there is still a significant concept-imbalance within them due to the long-tailed distribution of contexts (i.e., subject-object combinations). 
This concept-level imbalance poses a more pervasive and challenging issue compared to the predicate-level imbalance since subject-object pairs are inherently complex in combinations.
To address the issue, we propose \textbf{M}ulti-\textbf{C}oncept \textbf{L}earning (MCL), a nocel concept-level balanced learning framework orthogonal to existing SGG methods.
MCL first quantifies the concept-level imbalance across predicates in terms of different amounts of concepts, representing as multiple concept-prototypes within the same class.
It then effectively learns concept-prototypes by applying the Concept Regularization (CR) technique. 
Furthermore, to achieve balanced learning over different concepts, we introduce the \textcolor{red}{Concept-based Balanced Memory (\textcolor{red}{CBM})}, which guides SGG models to generate balanced representations for concept-prototypes.
Finally, we introduce a novel metric, mean Context Recall (mCR@K), as a complement to mean Recall (mR@K), to evaluate the model's performance across concepts (determined by contexts) within the same predicate. 
Extensive experiments demonstrate the remarkable efficacy of our model-agnostic strategy in enhancing the performance of benchmark models on both VG-SGG and OI-SGG datasets, leading to new state-of-the-art achievements in two key aspects: predicate-level unbiased relation recognition and concept-level compositional generability. 
<div align=center><img height="400" width="600" src=abstract.png></div>

## Framework
Within our Fine-Grained Predicates Learning (FGPL) framework, shown below, we first construct a Predicate Lattice concerning context information to understand ubiquitous correlations among predicates. Then, utilizing the Predicate Lattice, we develop a Category Discriminating Loss and an Entity Discriminating Loss which help SGG models differentiate hard-to-distinguish predicates.
<div align=center><img src=framework.png></div>

## Visualization
<div align=center><img  height="600" width="800" src=visual_sp-1.png></div>

## Device
Experiments of VG-SGG and OI-SGG are trained with an NVIDIA GeForce RTX 3090 GPU and 4 NVIDIA RTX A6000 GPUs, respectively.

## Dataset
Follow [DATASET.md](DATASET.md) for instructions of dataset preprocessing.

## Train
Follow the [instructions](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) to install and use the code. Also, we provide scripts for training models with FGPL our model (in `scripts/885train_[motif].sh`(https://github.com/XinyuLyu/G-USGG/blob/master/885train_trans.sh))\
    
## Test
The trained models(Motif-MCL) on Predcls\SGCLs\SGDet are released as below. 


| Predcls                                                                                                                                                           | SGCLs                                                                                                                                                            | SGDet                                                                                                                                                           |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Motif-MCL-Predcls](https://stduestceducn-my.sharepoint.com/:u:/g/personal/202011081621_std_uestc_edu_cn/EW0nkLFzPj9NsY2t0pAZNB8BU1YKaV2bOFBKBImf61N6Cw?e=vlp9pI) | [Motif-FMCL-SGCLS](https://stduestceducn-my.sharepoint.com/:u:/g/personal/202011081621_std_uestc_edu_cn/EcBLVZ7RD85OkyW5y4gVHUMB-kViQR7SOjHWNZXV6IiKeQ?e=QphPrF) | [Motif-MCL-SGDet](https://stduestceducn-my.sharepoint.com/:u:/g/personal/202011081621_std_uestc_edu_cn/EQuuuZMdxw9DriRPMjs5J0sBeGNa0NMXfGb-1ekQxYHx0Q?e=uMONJz) |

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
