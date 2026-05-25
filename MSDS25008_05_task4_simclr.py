import os
import torch
import random
import numpy as np
import torchvision
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
os.makedirs('results', exist_ok=True)
augmentations = T.Compose([
    T.RandomResizedCrop(32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])
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

     image = Image.fromarray(image)
     view1, view2 = self.transform(image)
     return view1, view2
base_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
loader = DataLoader(
    base_dataset,
    batch_size=128,
    shuffle=True
)
model = torchvision.models.resnet18(weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
feature_dim = 512
projection_dim = 128
model.fc = nn.Sequential(
    nn.Linear(512, feature_dim),
    nn.ReLU()
    nn.Linear(512, projection_dim)

)
model = model.to(device)
def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(device)
    similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
    positives = torch.cat([F.cosine_similarity(z1, z2), F.cosine_similarity(z2, z1)], dim=0)
    positves=positives.unsqueeze(1)
    logiits=torch.cat([positives, similarity_matrix], dim=1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long).to(device)
    logiits = logiits / temperature
    loss = F.cross_entropy(logiits, labels)
    return loss
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for views1, views2 in tqdm(loader):
        views1, views2 = views1.to(device), views2.to(device)
        optimizer.zero_grad()
        z1 = model(views1)
        z2 = model(views2)
        loss = nt_xent_loss(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() 
    avg_loss = total_loss / len(loader)
    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {avg_loss:.4f}"
    )
    encoder=model
    encoder.fc=nn.Identity()
    torch.save(encoder.state_dict(),"models/simclr_encoder.pt")

print("\nSimCLR encoder saved successfully.")