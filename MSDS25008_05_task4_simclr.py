import os
import torch
import random
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as dataloaders
from torch.utils.data import DataLoader, Subset
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

augmentations = T.Compose([
    T.RandomResizedCrop(32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])  
class SimCLRDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=None)
        self.aug = transform

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        view1 = self.aug(image)
        view2 = self.aug(image)
        return view1, view2
class SimCLREncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = torchvision.models.resnet18(weights=None)
        base.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc      = nn.Identity()  # outputs 512-d features
        self.encoder = base

    def forward(self, x):
        return self.encoder(x)  # (B, 512)

class SimCLRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SimCLREncoder()
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return z

def load_split_indices(path):
    with open(path, 'r') as f:
        return [int(line.strip()) for line in f]

ssl_indices = load_split_indices("splits/train_ssl_unlabeled.txt")

full_dataset = SimCLRDataset(root='./data', train=True, download=True, transform=augmentations)
ssl_subset   = Subset(full_dataset, ssl_indices)

loader = DataLoader(ssl_subset, batch_size=64, shuffle=True, drop_last=True)

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
model = SimCLRModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  
epochs = 50
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
torch.save(model.encoder.state_dict(), "models/simclr_encoder.pt")
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
# ===== FEATURE SIMILARITY AFTER TRAINING =====
print("\n" + "="*60)
print("Computing feature similarity after SimCLR training...")
print("="*60)

encoder.eval()
view1_test, view2_test = next(iter(loader))
view1_test, view2_test = view1_test.to(device), view2_test.to(device)

with torch.no_grad():
    features1_after = encoder(view1_test)
    features2_after = encoder(view2_test)

features1_after = F.normalize(features1_after, dim=1)
features2_after = F.normalize(features2_after, dim=1)

same_similarity_after = F.cosine_similarity(features1_after, features2_after)
same_similarity_after_mean = same_similarity_after.mean().item()

shuffled_indices = torch.randperm(features2_after.size(0))
shuffled_features2_after = features2_after[shuffled_indices]
different_similarity_after = F.cosine_similarity(features1_after, shuffled_features2_after)
different_similarity_after_mean = different_similarity_after.mean().item()

print(f"Same Image Two Views Similarity After: {same_similarity_after_mean:.4f}")
print(f"Different Images Similarity After:    {different_similarity_after_mean:.4f}")
print("="*60)


all_features_after = torch.cat([features1_after, features2_after], dim=0)
similarity_matrix_after = torch.mm(all_features_after, all_features_after.t()).cpu().numpy()

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(similarity_matrix_after, cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'Cosine Similarity'})
ax.set_title("Feature Similarity Matrix After SimCLR Training\n(Learned Encoder)", fontsize=12)
ax.set_xlabel("Feature Index")
ax.set_ylabel("Feature Index")

N = features1_after.shape[0]
ax.axhline(y=N, color='white', linewidth=2)
ax.axvline(x=N, color='white', linewidth=2)

plt.tight_layout()
plt.savefig("results/similarity_matrix_after_training.png", dpi=150, bbox_inches='tight')
plt.close()

print("✓ Similarity matrix after training saved!")

# ===== PRINT COMPARISON TABLE ==
print("\n" + "="*60)
print("SIMILARITY BEFORE vs AFTER SIMCLR TRAINING")
print("="*60)
print(f"{'Pair Type':<40} {'Before':<12} {'After':<12}")
print("-"*60)
same_before = 0.9890
diff_before = 0.9855
print(f"{'Same image, two augmented views':<40} {same_before:.4f}       {same_similarity_after_mean:.4f}")
print(f"{'Different images':<40} {diff_before:.4f}       {different_similarity_after_mean:.4f}")
print("="*60)
      