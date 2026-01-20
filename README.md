# Vision Transformer (ViT) from Scratch

A PyTorch implementation of Vision Transformer (ViT) built from scratch for MNIST digit classification. This project demonstrates the core concepts of Vision Transformers, including patch embedding, multi-head attention, and transformer encoders.

## 📋 Overview

This repository contains a complete implementation of a Vision Transformer model without using pre-built transformer libraries. The model is designed to classify handwritten digits from the MNIST dataset and serves as an educational resource for understanding the inner workings of Vision Transformers.

## 🏗️ Architecture

The Vision Transformer implementation consists of four main components:

### 1. **Patch Embedding**
- Converts input images into sequences of patches
- Uses 2D convolution to extract patches and embed them into feature vectors
- Patch size: 7×7 pixels from 28×28 MNIST images

### 2. **Transformer Encoder**
- Multi-head self-attention mechanism (2 attention heads)
- Layer normalization and residual connections
- MLP with GELU activation function
- 4 transformer encoder blocks

### 3. **Classification Head**
- Layer normalization followed by a linear classifier
- Maps the [CLS] token representation to 10 classes (digits 0-9)

### 4. **Positional Encoding**
- Learnable positional embeddings
- Added to patch embeddings to preserve spatial information

## 📁 Project Structure

```
├── ViT_scratch.py                    # Complete ViT implementation with training
├── coding ViT from scratch.ipynb     # Interactive Jupyter notebook
└── README.md                         # This file
```

## 🔧 Model Configuration

| Parameter | Value |
|-----------|-------|
| **Image Size** | 28×28 pixels |
| **Patch Size** | 7×7 pixels |
| **Number of Patches** | 16 patches |
| **Embedding Dimension** | 64 |
| **Attention Heads** | 2 |
| **Transformer Blocks** | 4 |
| **MLP Hidden Nodes** | 128 |
| **Number of Classes** | 10 |
| **Learning Rate** | 0.001 |
| **Batch Size** | 64 |
| **Training Epochs** | 5 |

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision
```

### Usage

#### Option 1: Run the Python Script
```bash
python ViT_scratch.py
```

#### Option 2: Use the Jupyter Notebook
Open `coding ViT from scratch.ipynb` in Jupyter Lab/Notebook for an interactive experience with step-by-step explanations.

### Training Output
The model will train for 5 epochs and display:
- Training loss per epoch
- Training accuracy per epoch
- Real-time progress updates

Example output:
```
Epoch 1/5
 Loss: 0.8234, Accuracy: 75.23%
Epoch 2/5
 Loss: 0.4567, Accuracy: 86.45%
...
```

## 🧠 Key Features

- **From Scratch Implementation**: No pre-built transformer libraries used
- **Educational Focus**: Clear, well-commented code for learning purposes
- **Modular Design**: Separate classes for each component
- **GPU Support**: Automatic device detection (CUDA/CPU)
- **MNIST Dataset**: Automatic download and preprocessing

## 📊 Model Components Breakdown

### Patch Embedding Layer
```python
class PatchEmbedding(nn.Module):
    def __init__(self):
        self.patch_embed = nn.Conv2d(1, 64, kernel_size=7, stride=7)
    
    def forward(self, x):
        x = self.patch_embed(x)      # Extract patches
        x = x.flatten(2)             # Flatten spatial dimensions  
        x = x.transpose(1, 2)        # (batch_size, num_patches, embedding_dim)
        return x
```

### Transformer Encoder Block
```python
class TransformerEncoder(nn.Module):
    def __init__(self):
        self.layer_norm1 = nn.LayerNorm(64)
        self.multihead_attention = nn.MultiheadAttention(64, 2, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(64)
        self.mlp = nn.Sequential(...)
    
    def forward(self, x):
        # Multi-head attention with residual connection
        residual1 = x
        x = self.layer_norm1(x)
        x = self.multihead_attention(x, x, x)[0]
        x = x + residual1
        
        # MLP with residual connection
        residual2 = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + residual2
        return x
```

## 🎯 Learning Objectives

This implementation helps understand:

1. **Vision Transformer Architecture**: How images are processed as sequences
2. **Patch Embedding**: Converting image patches to token embeddings
3. **Self-Attention Mechanism**: How attention works in transformers
4. **Positional Encoding**: Maintaining spatial relationships
5. **Classification Head**: Final prediction layer design

## 🔍 Technical Details

- **Input**: 28×28 grayscale MNIST images
- **Patches**: Each image is divided into 16 non-overlapping 7×7 patches
- **Sequence Length**: 17 tokens (16 patches + 1 CLS token)
- **Architecture**: Standard ViT with learnable positional embeddings
- **Training**: Cross-entropy loss with Adam optimizer

## 📈 Performance

The model achieves reasonable accuracy on MNIST classification within just 5 training epochs, demonstrating the effectiveness of the Vision Transformer architecture even for this simple implementation.

## 🤝 Contributing

Feel free to:
- Report bugs or issues
- Suggest improvements
- Add new features
- Improve documentation

## 📚 References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**Note**: This is an educational implementation focused on understanding Vision Transformers. For production use, consider using optimized libraries like `timm` or `transformers`.