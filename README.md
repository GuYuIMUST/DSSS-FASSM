## DSSS-FASSM

Official PyTorch implementation of:

**DARTS-searched feedback-augmented state space model for 3D pulmonary nodule classification**

*Computerized Medical Imaging and Graphics*, 135 (2026), 102820.  
[Paper](https://doi.org/10.1016/j.compmedimag.2026.102820)

## Overview

DSSS searches a compact 3D state-space architecture for pulmonary nodule
classification. The searched architecture is used to build FASSM and evaluated
on the LUNA16 dataset.

## Results

| Model  | Accu.  | Sens.  | Spec.  | F1 Score | Para. (M) | Batch size |
| ------ | ------: | ------: | ------: | --------: | --------: | ---------: |
| Fold-5 | 95.652 | 94.595 | 96.364 | 94.595 | 8.59 | 10 |
| Fold-6 | 91.304 | 92.857 | 90.411 | 88.636 | 8.59 | 10/4 |
| Fold-7 | 93.333 | 89.286 | 95.161 | 89.286 | 8.59 | 4 |
| Fold-8 | 91.346 | 91.837 | 90.909 | 90.909 | 8.59 | 4 |
| Fold-9 | 94.444 | 91.667 | 97.619 | 94.624 | 8.59 | 4 |

Five-fold mean: **93.22% accuracy**, **92.05% sensitivity**, **94.09%
specificity**, **91.61% F1 score**, and **0.944 ± 0.013 ROC AUC**.

## Installation

The code was checked with Python 3.10, PyTorch 2.1.1, CUDA 11.8, and Triton
2.1.0.

```bash
pip install -r requirements.txt
pip install -v ./kernels/selective_scan
```

## Data

Download LUNA16 and prepare 32×32×32 nodule crops:

```text
LUNA_ROOT/
├── cls/crop_v3/
└── rowfile/
    ├── subset0/
    ├── ...
    └── subset9/
```

Dataset files and trained checkpoints are not included.

## Usage

### Architecture search

Subset0 updates the architecture parameters, while subsets1–9 update the
network weights.

```bash
python master/train_search.py \
  --crop_dir /path/to/LUNA_ROOT/cls/crop_v3 \
  --luna_root /path/to/LUNA_ROOT/rowfile \
  --arch_subset 0 \
  --unrolled
```

### Training

Train one model for each held-out fold from 5 to 9:

```bash
python master/train_FASSM.py \
  --crop_dir /path/to/LUNA_ROOT/cls/crop_v3 \
  --luna_root /path/to/LUNA_ROOT/rowfile \
  --fold_validate 5 \
  --save fold5
```

### Evaluation

```bash
python master/test_Luna.py \
  --crop_dir /path/to/LUNA_ROOT/cls/crop_v3 \
  --luna_root /path/to/LUNA_ROOT/rowfile \
  --fold_test 5 \
  --pretrain_model_path /path/to/fold5/best_model.pth.tar
```

## Citation

```bibtex
@article{cui2026darts,
  title   = {DARTS-searched feedback-augmented state space model for 3D pulmonary nodule classification},
  author  = {Cui, Junjie and Gu, Yu and Wang, Mansheng and Chen, Meng and Yang, Lidong and Zhang, Baohua and Li, Jianjun and Liu, Xin and Hao, Juan and Tang, Siyuan and He, Qun},
  journal = {Computerized Medical Imaging and Graphics},
  volume  = {135},
  pages   = {102820},
  year    = {2026},
  doi     = {10.1016/j.compmedimag.2026.102820}
}
```

## Acknowledgement

This repository builds on
[DARTS](https://github.com/quark0/darts),
[Fair-DARTS](https://github.com/xiaomi-automl/FairDARTS),
[ProxylessNAS](https://github.com/MIT-HAN-LAB/ProxylessNAS),
[NAS-Lung](https://github.com/fei-hdu/NAS-Lung),
[DeepLung](https://github.com/uci-cbcl/DeepLung),
[VMamba](https://github.com/MzeroMiko/VMamba), and
[Mamba](https://github.com/state-spaces/mamba).
