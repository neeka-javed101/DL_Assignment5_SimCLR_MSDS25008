import os
import torch
import random
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from  tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
SEED=2026
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
os.makedirs("models", exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('graphs', exist_ok=True)
class SimCLRModel(nn.Module):
    def __init__(self, encoder, projection_dim=128):
        super().__init__()
        self.encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )
    
    def forward(self, x):
        features = self.encoder(x)  # 512-dim
        projections = self.projection_head(features)  # 128-dim
        return projections

augmentations = T.Compose([
    T.RandomResizedCrop(32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])
resnet = torchvision.models.resnet18(weights=None)
resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
resnet.maxpool = nn.Identity()
encoder = nn.Sequential(
    resnet.conv1,
    resnet.bn1,
    resnet.relu,
    resnet.layer1,
    resnet.layer2,
    resnet.layer3,
    resnet.layer4,
    resnet.avgpool,
    nn.Flatten()
)
model = SimCLRModel(encoder, projection_dim=128)
model = model.to(device)
class SimCLRDataset(torchvision.datasets.CIFAR10):

    def __init__(self, root, train=True,download=True, transform=None):

        super().__init__(root=root,train=train,download=download,transform=None)

        self.transform = transform

    
    def __getitem__(self, index):

     image = self.data[index]

     image = Image.fromarray(image)

     view1 = self.transform(image)

     view2 = self.transform(image)

     return view1, view2
base_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
dataset = SimCLRDataset(root='./data', train=True, download=True, transform=augmentations)
batch_size = 64  
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def nt_xent_loss(z1, z2, temperature=0.5):
    """NT-Xent loss from SimCLR paper"""
    batch_size = z1.size(0)
    
    # Normalize projections
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Concatenate: [2N, 128]
    representations = torch.cat([z1, z2], dim=0)
    
    # Cosine similarity matrix [2N, 2N]
    similarity_matrix = torch.mm(representations, representations.t())
    
    # Create positive pairs mask
    # Positive pairs: (i, i+N) and (i+N, i)
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(device)
    mask_pos = torch.zeros(2 * batch_size, 2 * batch_size, dtype=torch.bool).to(device)
    mask_pos[:batch_size, batch_size:] = torch.eye(batch_size).to(device)
    mask_pos[batch_size:, :batch_size] = torch.eye(batch_size).to(device)
    
    # Remove diagonal (self-similarity)
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
    
    # Extract positive similarities
    pos_sim = similarity_matrix[mask_pos].view(2 * batch_size, 1)
    
    # Normalize by temperature
    similarity_matrix = similarity_matrix / temperature
    pos_sim = pos_sim / temperature
    
    # Concatenate positive with all negatives
    logits = torch.cat([pos_sim, similarity_matrix], dim=1)
    
    # Labels: positive is at index 0
    labels = torch.zeros(2 * batch_size, dtype=torch.long).to(device)
    
    loss = F.cross_entropy(logits, labels)
    return loss
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
epochs = 30
train_losses = []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for views1, views2 in tqdm(loader):
        views1, views2 = views1.to(device), views2.to(device)
        optimizer.zero_grad()
        z1 = model(views1)
        z2 = model(views2)
        loss = nt_xent_loss(z1, z2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() 
    avg_loss = total_loss / len(loader)
    train_losses.append(avg_loss)  
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
    
encoder = model.encoder
torch.save(encoder.state_dict(), "models/simclr_encoder.pt")
print("✓ SimCLR encoder saved!")
plt.figure(figsize=(10, 6))
plt.plot(train_losses, linewidth=2)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("NT-Xent Loss", fontsize=12)
plt.title("SimCLR Pretraining Loss Curve", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("graphs/simclr_pretraining_loss.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Loss curve saved to graphs/simclr_pretraining_loss.png")
