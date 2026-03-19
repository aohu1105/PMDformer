# PMDFormer: Patch-Mean Decoupling Information Transformer for Long-Term Forecasting

[![Paper](https://img.shields.io/badge/Paper-OpenReview-blue?style=flat-square)](https://openreview.net/forum?id=rfJ41gK9Ct)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-brightgreen?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?style=flat-square)](https://pytorch.org/)

**English | [中文](README_CN.md)**

## Model Architecture

![PMDFormer Architecture](assets/Model.png)

## Main Results

![Main Results](assets/Result.png)

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Datasets

Download datasets from 📦 **[Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing)** and place them under `../dataset/` (parallel to this repo):

```
├── PMDformer/
└── dataset/
    ├── ETT-small/
    ├── weather/
    ├── electricity/
    ├── traffic/
    └── Solar/
```

### 3. Train

```bash
# Run all datasets at once
bash scripts/all.sh

# Or run a specific dataset
bash scripts/PMDformer/etth1.sh
```

## Citation

```bibtex
@inproceedings{
hu2026pmdformer,
title={{PMD}former: Patch-Mean Decoupling Transformer for Long-term Forecasting},
author={Ao Hu and Liangjian Wen and Jiang Duan and Yong Dai and Dongkai Wang and Jun Wang and YAN HE and Yukun Zhang and Ruoxi Jiang and Zenglin Xu},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=rfJ41gK9Ct}
}
```