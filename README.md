# Vision Transformer (ViT) — From Scratch

This repository walks through building a Vision Transformer (ViT) from scratch in a single interactive notebook. It’s ideal for learning the ViT architecture end-to-end, experimenting with hyperparameters, and understanding how patch embeddings, multi-head self-attention, and classification heads come together.

Key artifact: [coding ViT from scratch.ipynb](coding%20ViT%20from%20scratch.ipynb)

## Overview
- **Goal:** Implement and train a minimal yet complete ViT model, focusing on clarity and educational value.
- **Format:** A single, self-contained Jupyter/VS Code notebook.
- **Audience:** Students, practitioners, and anyone curious about transformer-based vision models.

## Features
- Patch extraction and linear embeddings for images
- Positional embeddings and transformer encoder blocks
- Multi-head self-attention with feed-forward layers
- Classification head and simple training loop
- Easy to modify hyperparameters and components for experiments

## Getting Started

### Prerequisites
- Windows, Linux, or macOS
- Python 3.9+ recommended
- VS Code (recommended) or Jupyter Notebook
- PyTorch (CPU or GPU)

### Create a virtual environment (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Install Jupyter and core packages

```powershell
pip install jupyter
pip install numpy matplotlib tqdm
```

### Install PyTorch
Follow the official instructions for your OS and CUDA/CPU setup:
https://pytorch.org/get-started/locally/

Example (CPU-only; version and index may differ):
```powershell
pip install torch torchvision torchaudio
```

## Run the Notebook

### In VS Code
- Open the repository folder in VS Code
- Open [coding ViT from scratch.ipynb](coding%20ViT%20from%20scratch.ipynb)
- Select a Python kernel (your `.venv`) and run cells top-to-bottom

### In Jupyter Notebook
```powershell
jupyter notebook
```
Then open [coding ViT from scratch.ipynb](coding%20ViT%20from%20scratch.ipynb) in your browser and run all cells.

## Usage & Workflow
1. Inspect model configuration cells (image size, patch size, embedding dim, number of heads, number of layers).
2. Prepare or load a dataset (e.g., CIFAR-10) per the notebook’s guidance.
3. Run the training loop to fit the ViT on your data.
4. Evaluate on a validation/test split and visualize results.

## Project Structure
- [coding ViT from scratch.ipynb](coding%20ViT%20from%20scratch.ipynb): Full implementation and training flow
- [README.md](README.md): You’re here

## Tips
- **GPU acceleration:** Ensure you installed a CUDA-enabled PyTorch build and have drivers configured; otherwise, training runs on CPU.
- **Reproducibility:** Set random seeds in the notebook when experimenting.
- **Experimentation:** Try different `patch_size`, `embed_dim`, `num_heads`, and `num_layers` to observe trade-offs.

## References
- Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,” 2020. https://arxiv.org/abs/2010.11929
- Official PyTorch docs: https://pytorch.org/docs/stable/index.html

## FAQ
- **Q: Which datasets are supported?**
  A: The notebook uses standard PyTorch dataset utilities (e.g., CIFAR-10). You can adapt to custom datasets by modifying the data loading section.
- **Q: Do I need GPUs?**
  A: No, but GPUs significantly speed up training. CPU works for small experiments.
- **Q: Can I export the model?**
  A: Yes. You can save `state_dict` and reload it later; see the notebook’s training section.

## Contributing
Feel free to open issues or pull requests with improvements, fixes, or additional examples.

## Acknowledgements
Thanks to the open-source community and researchers behind Vision Transformers for inspiration and guidance.

