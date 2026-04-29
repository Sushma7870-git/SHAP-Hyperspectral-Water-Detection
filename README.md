# SHAP-Driven Spectral Band Reduction for Efficient and Interpretable Hyperspectral Water Body Detection Using Deep CNN Architectures

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0+cu124-orange.svg)](https://pytorch.org/)


## Authors
Sushma Kumari, David Ayala-Cabrera, and Soumyabrata Dev

---

## Overview

This repository contains the official implementation of our paper:

> **SHAP-Driven Spectral Band Reduction for Efficient and Interpretable Hyperspectral Water Body Detection Using Deep CNN Architectures**

> Sushma Kumari, David Ayala-Cabrera, and Soumyabrata Dev


We propose a SHAP (SHapley Additive exPlanations)-driven framework for spectral band reduction in hyperspectral images, enabling efficient and interpretable water body detection using deep Convolutional Neural Network (CNN) architectures. By selecting only the most informative spectral bands using SHAP values, we significantly reduce computational cost while maintaining high detection accuracy.

---

## Project Structure

```
SHAP_Driven/
│
├── Datasets/               # Raw hyperspectral datasets
├── Hyperspectral/          # Preprocessed hyperspectral data
├── checkpoints/            # Saved model weights
├── metric_tables/          # Evaluation results and metrics
│
├── main.py                 # Main training and evaluation script
├── models.py               # CNN architecture definitions
├── datasets.py             # Dataset loading and preprocessing
├── custom_datasets.py      # Custom dataset class definitions
├── utils.py                # Utility functions (SHAP, metrics, visualization)
├── inference.py            # Model inference script
├── gt_split.py             # Binary Ground Truth Reformulation: Water vs Non-water
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Setup

### Prerequisites
Experiments were conducted on a workstation equipped with an AMD Ryzen 9 8945H CPU (8 cores, 16 threads,289
4.0 GHz), 32 GB RAM, and an NVIDIA GeForce RTX 4070 Laptop GPU running Windows 11. Python 3.10 with290
PyTorch 2.1.0+cu121 (CUDA 12.1) was used for GPU-accelerated training and inference

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Sushma7870-git/SHAP-Hyperspectral-Water-Detection.git

```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## Data

This project uses the following hyperspectral datasets:

| Dataset | Download Link |
|---------|--------------|
| Pavia Centre | [Download Pavia Centre Dataset](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |
| WHU-Hi-LongKou | [Download WHU-Hi-LongKou Dataset](https://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm) |

### Setting Up the Dataset Folder

1. Create a `Datasets/` folder in the project root:

```bash
mkdir Datasets
```

2. Download the datasets from the links above and place them in the `Datasets/` folder with the following structure:

```
Datasets/
├── PaviaC/
│   ├── Pavia.mat
│   └── Pavia_gt.mat
├── WHU-Hi-LongKou/
│   ├── WHU_Hi_LongKou.mat
│   └── WHU_Hi_LongKou_gt.mat
```

> **Note:** Make sure you have access to the datasets before running the project.

Binary Ground Truth Reformulation: Water vs Non-water, run:

```bash
python gt_split.py
```

---

## How to Run



### Visualization

This project uses [Visdom](https://github.com/fossasia/visdom) for real-time training visualization. Before running training, start the Visdom server in a separate terminal:

```bash
python -m visdom.server
```

Then open your browser and go to:
```
http://localhost:8097
```

Training metrics such as loss and accuracy will be displayed live during training.

---
### Training

```bash
python main.py --dataset PaviaC --model hamida --epoch 10 --cuda 0
```

**Arguments:**

| Argument | Description | Example |
|----------|-------------|---------|
| `--dataset` | Name of the dataset | `PaviaC` |
| `--model` | CNN model architecture | `hamida` |
| `--epoch` | Number of training epochs | `10` |
| `--cuda` | GPU device ID | `0` |

---

## Results

Evaluation metrics including Accuracy, Precision, Recall, and Training Time are saved in the `metric_tables/` folder after training.

| Metric        | Before SHAP (All Bands) | After SHAP (Reduced Bands) |
|---------------|------------------------|---------------------------|
| Accuracy      | ...                    | ...                       |
| Precision     | ...                    | ...                       |
| Recall        | ...                    | ...                       |
| Training Time | ...                    | ...                       |
| Peak GPU Mem  | ...                    | ...                       |

---

## Requirements

Key dependencies (see `requirements.txt` for full list):

- `torch==2.6.0+cu124`
- `torchvision==0.21.0+cu124`
- `numpy==2.2.2`
- `scikit-learn==1.6.1`
- `scikit-image==0.25.1`
- `matplotlib==3.10.0`
- `pandas==2.2.3`
- `spectral==0.19`
- `tifffile==2025.1.10`

---

## Reference

If you use this code in your research, please cite our paper:

```bibtex
@misc{kumari2025shap,
  title={SHAP-Driven Spectral Band Reduction for Efficient and Interpretable 
         Hyperspectral Water Body Detection Using Deep CNN Architectures},
  author={Kumari, Sushma and Ayala-Cabrera, David and Dev, Soumyabrata},
  year={2025},
  note={Manuscript under review}
}
```
This work builds upon the following:

```bibtex
@article{audebert2019deep,
  title={Deep learning for classification of hyperspectral data: A comparative review},
  author={Audebert, Nicolas and Le Saux, Bertrand and Lef{\`e}vre, S{\'e}bastien},
  journal={IEEE geoscience and remote sensing magazine},
  volume={7},
  number={2},
  pages={159--173},
  year={2019},
  publisher={IEEE}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact:
- **Sushma Kumari** — [sushma.kumari@ucdconnect.ie] 
