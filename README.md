# LCCS

This repository contains code demonstrating the method in our IJCAI 2022 paper [Few-Shot Adaptation of Pre-Trained Networks for Domain Shift](https://www.ijcai.org/proceedings/2022/0232.pdf), and [arXiv version](https://arxiv.org/pdf/2205.15234.pdf) containing both main manuscript and appendix.

## Setting up

### Prerequisites

This code makes use of [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). Please follow the instructions at https://github.com/KaiyangZhou/Dassl.pytorch#installation to install `dassl`.

We used NVIDIA container image for PyTorch, release 20.12, to run experiments.

### Dataset

This demonstration runs on [PACS](https://drive.google.com/uc?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWh). Please download the dataset and save in `lccs/imcls/data/`.

## Training and Evaluation

#### 1. Training source models
From `lccs/imcls/scripts/`, run `./run_source.sh`.
The source models will be saved in `lccs/imcls/output_source_models/`.

#### 2. Adapt source models with LCCS
From `lccs/imcls/scripts/`, run `./run_lccs.sh`.
The outputs after adaptation will be saved in `lccs/imcls/output_results/`.

#### 3. Summarize model performance
From `lccs/imcls/results_scripts/`, first run `./collect_results.sh` and then `./consolidate_results.sh`.

## Citation
```
@inproceedings{zhang2022lccs,
  title     = {Few-Shot Adaptation of Pre-Trained Networks for Domain Shift},
  author    = {Zhang, Wenyu and Shen, Li and Zhang, Wanyue and Foo, Chuan-Sheng},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Lud De Raedt},
  pages     = {1665--1671},
  year      = {2022},
  month     = {7},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2022/232},
  url       = {https://doi.org/10.24963/ijcai.2022/232},
}
```

## Acknowledgements

Our implementation is based off the repository [MixStyle](https://github.com/KaiyangZhou/mixstyle-release). Thanks to the MixStyle implementation.