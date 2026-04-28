# SHAP-Driven Spectral Band Reduction for Efficient and Interpretable Hyperspectral Water Body Detection Using Deep CNN Architectures

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0+cu124-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Authors
Sushma Kumari, David Ayala-Cabrera, and Soumyabrata Dev

---

## Overview

This repository contains the official implementation of our paper:

> **SHAP-Driven Spectral Band Reduction for Efficient and Interpretable Hyperspectral Water Body Detection Using Deep CNN Architectures**
> Sushma Kumari, David Ayala-Cabrera, and Soumyabrata Dev
> *Journal Name*, Year. DOI: `[to be added]`

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
├── gt_split.py             # Ground truth train/test splitting
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Setup

### Prerequisites
- Python 3.x
- CUDA 12.4
- NVIDIA GPU (recommended)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/SHAP_Driven.git
cd SHAP_Driven
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

To split ground truth into training and testing sets, run:

```bash
python gt_split.py
```

---

## How to Run

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

### Inference

To run inference on a trained model:

```bash
python inference.py --checkpoint checkpoints/[model_checkpoint].pth
```

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

If you use this code in your research, please cite:

```bibtex
@article{kumari2025shap,
  title     = {SHAP-Driven Spectral Band Reduction for Efficient and Interpretable Hyperspectral Water Body Detection Using Deep CNN Architectures},
  author    = {Kumari, Sushma and Ayala-Cabrera, David and Dev, Soumyabrata},
  journal   = {Journal Name},
  year      = {2025},
  doi       = {to be added}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or issues, please open a GitHub issue or contact:
- **Sushma Kumari** — [email@domain.com] | [phone number]
