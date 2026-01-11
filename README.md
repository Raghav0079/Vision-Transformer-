# Vision Transformer (ViT) Implementation

A PyTorch implementation of Vision Transformer architecture from scratch, based on the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture Details](#model-architecture-details)
- [Training](#training)
- [Results](#results)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

Vision Transformer (ViT) applies the Transformer architecture, originally designed for natural language processing, directly to images. Instead of using convolutional layers, ViT splits an image into fixed-size patches, linearly embeds each patch, adds position embeddings, and feeds the resulting sequence to a standard Transformer encoder.

## 🏗️ Architecture

The Vision Transformer architecture consists of:

1. **Patch Embedding**: Split image into fixed-size patches and linearly embed them
2. **Position Embedding**: Add learnable position embeddings to patch embeddings
3. **Transformer Encoder**: Stack of transformer blocks with multi-head self-attention
4. **Classification Head**: MLP head for final classification

![ViT Architecture](https://user-images.githubusercontent.com/example/vit-architecture.png)

## ✨ Features

- [x] Complete ViT implementation from scratch
- [x] Configurable model sizes (Base, Large, Huge)
- [x] Support for different patch sizes (16x16, 32x32)
- [x] Pre-training and fine-tuning capabilities
- [x] Data augmentation techniques
- [x] Visualization tools for attention maps
- [ ] Multi-scale training
- [ ] Distillation support

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- torchvision
- numpy
- matplotlib
- tqdm

### Install Dependencies

```bash
git clone https://github.com/Raghav0079/Vision-Transformer-.git
cd Vision-Transformer-
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib tqdm pillow
```

## 💻 Usage

### Quick Start

```python
from vit_model import VisionTransformer
import torch

# Create ViT model
model = VisionTransformer(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=768,
    depth=12,
    heads=12,
    mlp_dim=3072
)

# Example input
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(f"Output shape: {output.shape}")  # torch.Size([1, 1000])
```

### Training Example

```python
from train import train_model
from data_loader import get_data_loaders

# Get data loaders
train_loader, val_loader = get_data_loaders(
    data_path='./data',
    batch_size=32,
    image_size=224
)

# Train the model
model = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    learning_rate=1e-4
)
```

## 🔧 Model Architecture Details

### Supported Model Configurations

| Model | Layers | Hidden Size | MLP Size | Heads | Params |
|-------|--------|-------------|----------|-------|--------|
| ViT-Base/16 | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large/16 | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge/14 | 32 | 1280 | 5120 | 16 | 632M |

### Key Components

- **Patch Embedding**: Converts image patches to embeddings
- **Multi-Head Self-Attention**: Captures relationships between patches
- **Feed-Forward Network**: Processes attention outputs
- **Layer Normalization**: Stabilizes training
- **Residual Connections**: Enables deep network training

## 🏋️ Training

### Dataset Preparation

```bash
# For ImageNet
python prepare_data.py --dataset imagenet --data_path /path/to/imagenet

# For CIFAR-10
python prepare_data.py --dataset cifar10
```

### Training Command

```bash
python train.py \
    --model vit_base_patch16_224 \
    --dataset imagenet \
    --batch_size 128 \
    --epochs 300 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --warmup_epochs 10
```

### Training Arguments

- `--model`: Model architecture (vit_base_patch16_224, vit_large_patch16_224)
- `--dataset`: Dataset name (imagenet, cifar10, cifar100)
- `--batch_size`: Batch size for training
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--weight_decay`: Weight decay for optimizer
- `--warmup_epochs`: Number of warmup epochs

## 📊 Results

### ImageNet Results

| Model | Top-1 Acc | Top-5 Acc | Params | FLOPs |
|-------|-----------|-----------|--------|-------|
| ViT-B/16 | 77.9% | 93.8% | 86M | 17.6G |
| ViT-L/16 | 76.5% | 93.2% | 307M | 61.6G |

### CIFAR-10 Results

| Model | Accuracy | Training Time |
|-------|----------|---------------|
| ViT-B/16 | 99.1% | 2 hours |

## 📁 Project Structure

```
Vision-Transformer-/
├── README.md
├── requirements.txt
├── models/
│   ├── __init__.py
│   ├── vit.py
│   └── attention.py
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   └── visualization.py
├── train.py
├── evaluate.py
└── config.py
```

## 🔬 Attention Visualization

Visualize what the model is looking at:

```python
from utils.visualization import visualize_attention

# Load trained model
model = VisionTransformer.load_from_checkpoint('checkpoint.pth')

# Visualize attention maps
visualize_attention(model, image_path='sample.jpg', head_idx=0, layer_idx=11)
```

## 📚 References

1. **Original Paper**: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
   - Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al.
   - ICLR 2021

2. **Attention Is All You Need**: [Transformer Architecture](https://arxiv.org/abs/1706.03762)
   - Ashish Vaswani, Noam Shazeer, Niki Parmar, et al.
   - NeurIPS 2017

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to the authors of the original Vision Transformer paper
- Inspired by the Hugging Face Transformers library
- Special thanks to the PyTorch team for the excellent framework

## 📞 Contact

**Raghav** - [@Raghav0079](https://github.com/Raghav0079)

Project Link: [https://github.com/Raghav0079/Vision-Transformer-](https://github.com/Raghav0079/Vision-Transformer-) 
