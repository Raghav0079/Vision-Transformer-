"""
Vision Transformer (ViT) Implementation from Scratch
====================================================
This script implements a Vision Transformer model using PyTorch for MNIST classification.
"""

# importing necessary libraries
import torch
import torchvision
import torch.utils.data as dataloader
import torch.nn as nn

from torchvision import datasets, transforms

# import dataset
# Define transforms
transform = transforms.Compose([transforms.ToTensor()])

# Download and load MNIST dataset
mnist_train = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
mnist_test = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
mnist_val = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
transform = torchvision.transforms.Compose([transforms.ToTensor()])

# define batches
train_loader = dataloader.DataLoader(mnist_train, batch_size=64, shuffle=True)
val_loader = dataloader.DataLoader(mnist_val, batch_size=64, shuffle=True)

# define variable
num_classes = 10
batch_size = 64
num_channels = 1
image_size = 28
patch_size = 7
num_patches = (image_size // patch_size) ** 2
embedding_dim = 64
attention_heads = 2
transformer_blocks = 4
learning_rate = 0.001
epochs = 5
mlp_hidden_nodes = 128

# sample a data point from train_loader
sample_data = next(iter(train_loader))
images, labels = sample_data
print(f"Image batch shape: {images.size()}")
print(f"Label batch shape: {labels.size()}")

# Creating Patches and Embedding them
patch_embed = nn.Conv2d(
    num_channels, embedding_dim, kernel_size=patch_size, stride=patch_size
)
print(patch_embed(images).shape)

# Flatten the patches
patch_embed_output_flatten = patch_embed(images).flatten(2)
print(patch_embed_output_flatten.shape)
print(
    patch_embed_output_flatten.transpose(1, 2).shape
)  # Expected output: (batch_size, num_patches, embedding_dim)

### part 1 : patch embedding
### part 2 : transformer encoder
### part 3: mlp head
### transformer class

# =============================================================================
# Part 1: Patch Embedding
# =============================================================================


# Patch Embedding Module
class PatchEmbedding(nn.Module):
    # initialize the module
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            num_channels, embedding_dim, kernel_size=patch_size, stride=patch_size
        )

    # forward method
    def forward(self, x):
        # patch embedding
        x = self.patch_embed(x)
        # flatten the patches
        x = x.flatten(2)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embedding_dim)
        return x


# =============================================================================
# Part 2: Transformer Encoder
# =============================================================================


# Transformer Encoder Module
class TransformerEncoder(nn.Module):
    def __init__(self):  # initialize the module
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.multihead_attention = nn.MultiheadAttention(
            embedding_dim, attention_heads, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden_nodes),
            nn.GELU(),
            nn.Linear(mlp_hidden_nodes, embedding_dim),
        )

    # forward method
    def forward(self, x):
        residual1 = x
        x = self.layer_norm1(x)
        x = self.multihead_attention(x, x, x)[0]
        x = x + residual1

        residual2 = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + residual2

        return x


# =============================================================================
# Part 3: MLP Head
# =============================================================================


# MLP Head Module
class MLP_head(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.mlp_head = nn.Linear(embedding_dim, num_classes)

    # forward method
    def forward(self, x):
        x = self.layer_norm1(x)
        x = self.mlp_head(x)

        return x


# =============================================================================
# Part 4: Vision Transformer Model
# =============================================================================


# Vision Transformer Module
class VisionTransformer(nn.Module):
    def __init__(self):  # initialize the module
        super().__init__()
        self.patch_embedding = PatchEmbedding()
        self.clas_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, embedding_dim)
        )
        self.transformer_blocks = nn.Sequential(
            *[TransformerEncoder() for _ in range(transformer_blocks)]
        )
        self.mlp_head = MLP_head()

    # forward method
    def forward(self, x):
        x = self.patch_embedding(x)
        B = x.size(0)
        class_token = self.clas_token.expand(B, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = x + self.position_embedding
        x = self.transformer_blocks(x)
        x = x[:, 0]
        x = self.mlp_head(x)

        return x


# =============================================================================
# Training Setup and Loop
# =============================================================================

# Define device, model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct_epoch = 0
    total_epoch = 0

    print(f"Epoch {epoch+1}/{epochs}")

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_epoch += labels.size(0)
        correct_epoch += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct_epoch / total_epoch
    print(f" Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
