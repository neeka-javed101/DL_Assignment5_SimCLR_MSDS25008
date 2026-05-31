import os
import torch
import random
import numpy as np
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
import torch.nn as nn
import seaborn as sns
from PIL import Image
# Set random seeds for reproducibility
SEED=2026
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(42)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
os.makedirs('results', exist_ok=True)
augmentations = T.Compose([
    T.RandomResizedCrop(32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])
# Function to load indices from text files
class TwoViewsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform
    def __call__(self, x):
       view1 = self.base_transform(x)
       view2 = self.base_transform(x)
       return view1, view2
class SimCLRDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, download=True, transform=None):
        # ✅ Pass transform=None to parent, we'll handle it ourselves
        super().__init__(root=root, train=train, download=download, transform=None)
        self.transform = transform
    
    def __getitem__(self, index):

     image = self.data[index]
     label = self.targets[index]
     image = Image.fromarray(image)
     view1, view2 = self.transform(image)
     return view1, view2, label
base_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
two_view_transform = TwoViewsTransform(augmentations)
dataset = SimCLRDataset(root='./data', train=True, download=True, transform=two_view_transform)
dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
# Build a random encoder (ResNet-18 with random weights)
model = torchvision.models.resnet18(weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Identity()   
for name, module in model.named_modules():
    if isinstance(module, nn.BatchNorm2d):
        # reset to identity-like behavior
        module.weight.data.fill_(1)
        module.bias.data.fill_(0)
        module.running_mean.fill_(0)
        module.running_var.fill_(1)

model = model.to(device)
model.eval()
# Extract features for two views of the same images

view1, view2, labels = next(iter(dataloader))
view1, view2 = view1.to(device), view2.to(device)
with torch.no_grad():
    features1 = model(view1)
    features2 = model(view2)
print(f"features1 shape: {features1.shape}")
print(f"features1 sample values: {features1[0][:5]}")
print(f"features1 norm before normalize: {features1.norm(dim=1).mean():.4f}")
features1 = F.normalize(features1, dim=1)
features2 = F.normalize(features2, dim=1)
same_similarity =F.cosine_similarity(features1, features2)
same_similarity_mean = same_similarity.mean().item()
torch.manual_seed(SEED)
shuffled_indices = torch.randperm(features2.size(0))

shuffled_features2 = features2[shuffled_indices]
different_similarity = F.cosine_similarity(features1, shuffled_features2)
different_similarity_mean = different_similarity.mean().item()
print("\n" + "="*60)
print("FEATURE SIMILARITY BEFORE TRAINING (Random Model)")
print("="*60)
print(f"Same Image Two Views Similarity: {same_similarity_mean:.4f}")
print(f"Different Images Similarity: {different_similarity_mean:.4f}")
print(f"Difference: {same_similarity_mean - different_similarity_mean:.4f}")
print("="*60)

all_features = torch.cat([features1, features2], dim=0)
similarity_matrix = torch.mm(all_features, all_features.t()).cpu().numpy()

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(similarity_matrix, cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'Cosine Similarity'})
ax.set_title("Feature Similarity Matrix Before Training\n(Random Encoder)", fontsize=12)
ax.set_xlabel("Feature Index")
ax.set_ylabel("Feature Index")

N = features1.shape[0]
ax.axhline(y=N, color='white', linewidth=2)
ax.axvline(x=N, color='white', linewidth=2)

plt.tight_layout()
plt.savefig("results/similarity_matrix_before_training.png", dpi=150, bbox_inches='tight')
plt.close()

print("\n✓ Similarity matrix heatmap saved!")
print("✓ Saved: results/similarity_matrix_before_training.png")