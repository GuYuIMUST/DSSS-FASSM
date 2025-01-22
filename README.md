# DSSS-FASSM

## Introduction

**Feedback-Augmented State Space Model in DARTS-Searched Mamba variant Architecture for Automatic 3D Pulmonary Nodule Classification**

Sorry, the paper is currently under review. Once the paper is published, we will release the code.

## Results

### Feedback-Augmented State Space Model (FASSM)

| model               | Accu.       | Sens. | Spec. | F1 Score | para.(M)    |
| ------------------- | ----------- | ----- | ----- | -------- | ----------- |
| Multi-crop CNN      | 87.14       | -     | -     | -        | -           |
| Nodule-level 2D CNN | 87.30       | 88.50 | 86.00 | 87.23    | -           |
| Vanilla 3D CNN      | 87.40       | 89.40 | 85.20 | 87.25    | -           |
| ADNN      | 90.11       | - | - | -    | -           |
| DeepLung            | 90.44       | -        | -     | -        | 141.57      |
| AE-DPN              | 90.24       | 92.04 | 88.94 | 90.45    | 678.69      |
| NASLung             | 90.77       | 85.37 | 95.04 | 89.29    | 16.84       |
| NAS-qa             | 90.85       | 86.04 | 91.02 | 88.89    | -       |
| **FASSM(ours)** | **93.22(top)** | **92.05(top)** | 94.09(second) | **91.61(top)** | **8.59(top)** |

### Verification results

Due to limited experimental conditions, the same hyperparameters were used and we did not conduct particularly fine tuning. Therefore, the experimental results may be further improved.

| Model  | Accu.  | Sens.  | Spec.  | F1 Score | para. |batchsize |
| ------ | ------ | ------ | ------ | -------- | ----- |-----  |
| Fold-5 | 95.652 | 94.595 | 96.364 | 94.595   | 8.59  |10 |
| Fold-6 | 91.304 | 92.857 | 90.411 | 88.636   | 8.59  |10/4 |
| Fold-7 | 93.333 | 89.286 | 95.161 | 89.286   | 8.59  |4 |
| Fold-8 | 91.346 | 91.837 | 90.909 | 90.909   | 8.59  |4 |
| Fold-9 | 94.444 | 91.667 | 97.619 | 94.624 | 8.59  |4 |

## Usage
To run our code, you only need one GeForce RTX 4090(24G memory).

#### Preprocessing
You need to download the LUNA16 dataset by yourself and adjust the corresponding paths.
```
prepare.py \\
```
```
nodclsgbt.py \\
```
#### Search on the LUNA16 dataset（Differentiable State Space Supergraph (DSSS)）

The number of pre-training epochs W in the program is the number of epochs used to solely train W. The epoch when the validation set first reaches its optimal value needs to be determined again if you plan to use your own dataset, as detailed in section 4.1 of the paper.

```
train_search.py \\
```
#### Evaluation on the LUNA16 dataset（Feedback-Augmented State Space Model (FASSM)）

Some randomness is introduced in certain aspects of the code, such as partial channel connections, to allow for fluctuations in results. Due to the 37-hour training time required for the model, we did not finely tune the hyperparameters, and thus the experimental results we present may not represent the optimal values.

```
python train_FASSM.py \\
```

### Requirements
operating system
- Linux
To ensure the code can run, we provide versions of some libraries.
- python-3.7.13
- numpy-1.21.5
- pytorch-1.21.1
- pandas-1.3.5
- opencv-python-4.8.1

## Acknowledgement 

If there are any missing citations, please contact us. It is an unintentional omission, and we will add the citations accordingly.

 **This code is based on the implementation of  [DARTS](https://github.com/quark0/darts)，[Fair-DARTS](https://github.com/xiaomi-automl/FairDARTS)，[ProxylessNAS](https://github.com/MIT-HAN-LAB/ProxylessNAS)，[NAS-Lung](https://github.com/fei-hdu/NAS-Lung)，[DeepLung](https://github.com/uci-cbcl/DeepLung), [VMamba](*https://github.com/MzeroMiko/VMamba*) and [Mamba](*https://github.com/state-spaces/mamba*).**

## Selected References

If there are any missing citations, please contact us. It is an unintentional omission, and we will add the citations accordingly.

- W. Zhu, C. Liu, W. Fan, X. Xie, Deeplung: Deep 3d dual path nets for automated pulmonary nodule detection and classification,  2018 IEEE winter conference on applications of computer vision (WACV), IEEE2018, pp. 673-681.
- H. Liu, K. Simonyan, Y. Yang, Darts: Differentiable architecture search, arXiv preprint arXiv:1806.09055, (2018).
- H. Jiang, F. Shen, F. Gao, W. Han, Learning efficient, explainable and discriminative representations for pulmonary nodules classification, Pattern Recognition, 113 (2021) 107825.
- X. Chu, T. Zhou, B. Zhang, J. Li, Fair darts: Eliminating unfair advantages in differentiable architecture search,  European conference on computer vision, Springer2020, pp. 465-480.
- S.G. Armato III, G. McLennan, L. Bidaut, M.F. McNitt‐Gray, C.R. Meyer, A.P. Reeves, B. Zhao, D.R. Aberle, C.I. Henschke, E.A. Hoffman, The lung image database consortium (LIDC) and image database resource initiative (IDRI): a completed reference database of lung nodules on CT scans, Medical physics, 38 (2011) 915-931.
- K. Kuan, M. Ravaut, G. Manek, H. Chen, J. Lin, B. Nazir, C. Chen, T.C. Howe, Z. Zeng, V. Chandrasekhar, Deep learning for lung cancer detection: tackling the kaggle data science bowl 2017 challenge, arXiv preprint arXiv:1705.09435, (2017).
- Liu, Y., Tian, Y., Zhao, Y., Yu, H., Xie, L., Wang, Y., Ye, Q., & Liu, Y. (2024). VMamba: Visual State Space Model. ArXiv, abs/2401.10166.
- Gu, A. and Dao, T., 2023. Mamba: Linear-time sequence modeling with selective state spaces. *arxiv preprint arxiv:2312.00752*.

