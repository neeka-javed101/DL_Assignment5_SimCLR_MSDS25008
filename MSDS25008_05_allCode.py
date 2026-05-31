# MSDS25008_05_allCode.py
# MSDS25008_05_task1_supervised.py
import os
import torch
import torch.nn as nn
import random 
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader,Subset
import torchvision.transforms as T
# Set random seeds for reproducibility
SEED=2026
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
os.makedirs('graphs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
def load_split_indices(path):
    with open(path, 'r') as f:
        indices = [int(line.strip()) for line in f.readlines()]
    return indices
train_transforms = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])
test_transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])
# Load datasets and create dataloaders for the specified splits
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)
train_indices = load_split_indices("splits/train_labeled_10percent.txt")
val_indices = load_split_indices("splits/val.txt")
test_indices = load_split_indices("splits/test.txt")
train_subset = Subset(train_dataset, train_indices)
test_subset =  Subset(test_dataset, test_indices)
val_subset =   Subset(train_dataset, val_indices)
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
model = torchvision.models.resnet18(weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(512, 10)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
# Training loop with early stopping
num_epochs = 30
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
best_val_accuracy = 0
patience = 3
patience_counter = 0

    
    # Forward pass and loss computation
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [TRAIN]"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
    train_loss = running_loss / total
    train_acc = 100 * correct / total
    train_losses.append(train_loss) 
    train_accuracies.append(train_acc)

    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
# Validation loop
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [VAL]"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_running_loss / val_total
    val_accuracy = 100 * val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Epoch [{epoch+1:3d}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
    # Early stopping and model checkpoint check
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        patience_counter = 0  
        torch.save(model.state_dict(), 'models/supervised_best.pt')
        print(f"  → Best model saved!")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
plt.figure(figsize=(8,5))

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Supervised Training Loss")

plt.legend()

plt.savefig(
    "graphs/supervised_loss.png"
)

plt.close()
model.load_state_dict(
    torch.load("models/supervised_best.pt")
)

model.eval()

all_labels = []
all_preds = []

correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()

        total += labels.size(0)

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_preds.extend(
            preds.cpu().numpy()
        )


test_acc = 100 * correct / total

print(f"\nFinal Test Accuracy: {test_acc:.2f}%")


 #CONFUSION MATRIX 

cm = confusion_matrix(
    all_labels,
    all_preds
)

disp = ConfusionMatrixDisplay(cm)

fig, ax = plt.subplots(figsize=(8,8))

disp.plot(ax=ax)

plt.title("Supervised Confusion Matrix")

plt.savefig(
    "results/supervised_confusion_matrix.png"
)

plt.close()
# MSDS25008_05_augmentation.py
import os
import random
import torch
import numpy as np
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
# Set random seeds for reproducibility
SEED=2026
random.seed(SEED)
np.random.seed(SEED)
os.makedirs("results", exist_ok=True)
simclr_transforms = T.Compose([
    T.RandomResizedCrop(32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),  
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616))
])
# Function to load indices from text files
class TwoViewTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform
    
    def __call__(self, x):
        view1 = self.base_transform(x)
        view2 = self.base_transform(x)
        return view1, view2
two_view_transform = TwoViewTransform(simclr_transforms)
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=two_view_transform)

fig, axes = plt.subplots(
    10,
    3,
    figsize=(9, 30)
)
# Visualize original and augmented views for 10 random samples
raw_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True, 
    transform=None  
)
def tensor_to_image(tensor):
    """Convert normalized tensor [C, H, W] to numpy image [H, W, C]"""
    #  Denormalize (reverse the Normalize transform)
    denormalize = T.Compose([
        T.Normalize(mean=[-0.4914/0.2470, -0.4822/0.2435, -0.4465/0.2616],
                    std=[1/0.2470, 1/0.2435, 1/0.2616])
    ])
    
    tensor = denormalize(tensor)
    tensor = torch.clamp(tensor, 0, 1)  # Clamp to valid range
    img = tensor.permute(1, 2, 0).numpy()  # Convert to [H, W, C]
    return img

fig, axes = plt.subplots(10, 3, figsize=(9, 30))

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print("Generating augmentation examples...")
# Loop through 10 random samples and visualize original + two augmented views
for i in range(10):
    raw_image, label = raw_dataset[i] 
    view1, view2 = two_view_transform(raw_image)  

    axes[i,0].imshow(raw_image)

    axes[i,0].set_title("Original")

    axes[i,0].axis("off")

    # VIEW 1

    axes[i,1].imshow(tensor_to_image(view1))

    axes[i,1].set_title("Augmented View 1")

    axes[i,1].axis("off")

    # VIEW 2

    axes[i,2].imshow(tensor_to_image(view2)) 

    axes[i,2].set_title("Augmented View 2")

    axes[i,2].axis("off")


plt.tight_layout()

plt.savefig(
    "results/augmentation_examples.png"
)

plt.close()


print("\nAugmentation visualization saved successfully.")
# MSDS25008_05_task3_similarity.py
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
# MSDS25008_05_task4_simclr.py
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
import torch.utils.data as dataloader
from torch.utils.data import DataLoader, Subset
from  tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
# Set random seeds for reproducibility
SEED=2026
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
os.makedirs("models", exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('graphs', exist_ok=True)
# Function to load indices from text files

augmentations = T.Compose([
    T.RandomResizedCrop(32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])  
# Data preparation and similarity analysis before training
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
    # simCLR training loop and evaluation

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
# Train SimCLR modeland save encoder weights
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
backbone = model.encoder.encoder
torch.save(backbone.state_dict(), "models/simclr_encoder.pt")

print("✓ SimCLR encoder saved!")
# Plot training loss curve
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
# Set backbone to evaluation modeand compute features for a batch of test images

backbone.eval()
view1_test, view2_test = next(iter(loader))
view1_test, view2_test = view1_test.to(device), view2_test.to(device)

with torch.no_grad():
    features1_after = backbone(view1_test)
    features2_after = backbone(view2_test)

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
# Generate similarity matrix heatmap after training

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
# MSDS25008_05_task5_linear_probe.py
import os
import torch
import random
import numpy as np
import torchvision
import torchvision.transforms as T
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Subset
SEED=2026
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
os.makedirs('graphs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
def load_split_indices(file_path):
    with open(file_path, 'r') as f:
        indices = [int(line.strip()) for line in f]
    return indices
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
               std=(0.2470, 0.2435, 0.2616))
])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
val_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_indices = load_split_indices("splits/train_labeled_10percent.txt")

val_indices = load_split_indices("splits/val.txt")
test_indices = load_split_indices(    "splits/test.txt")
train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=64, shuffle=True)
val_loader = DataLoader(Subset(val_dataset, val_indices), batch_size=64, shuffle=False)
test_loader = DataLoader(Subset(test_dataset, test_indices), batch_size=64, shuffle=False)
train_accuracies, val_accuracies = [], []
# ===== EXPERIMENT A: RANDOM ENCODER LINEAR PROBE =====
print("\n" + "="*50)
print("EXPERIMENT A: RANDOM ENCODER LINEAR PROBE")
print("="*50)

random_encoder = torchvision.models.resnet18(weights=None)
random_encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
random_encoder.maxpool = nn.Identity()
random_encoder.fc = nn.Identity()
random_encoder.to(device)
for param in random_encoder.parameters():
    param.requires_grad = False

random_classifier = nn.Linear(512, 10).to(device)
random_optimizer = torch.optim.Adam(random_classifier.parameters(), lr=3e-4)
best_random_val_acc = 0.0

for epoch in range(20):
    random_classifier.train()
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader, desc=f"[Random] Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            features = random_encoder(images)
        outputs = random_classifier(features)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        random_optimizer.zero_grad()
        loss.backward()
        random_optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_acc = 100 * correct / total
    random_classifier.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            features = random_encoder(images)
            outputs = random_classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_acc = 100 * val_correct / val_total
    print(f'[Random] Epoch [{epoch+1}/20] Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%')

    if val_acc > best_random_val_acc:
        best_random_val_acc = val_acc
        torch.save(random_classifier.state_dict(), "models/best_random_classifier.pt")
random_classifier.load_state_dict(torch.load("models/best_random_classifier.pt"))
random_classifier.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        features = random_encoder(images)
        outputs = random_classifier(features)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
random_test_acc = 100 * correct / total
print(f'\nRandom Encoder Test Accuracy: {random_test_acc:.2f}%')
encoder=torchvision.models.resnet18(weights=None)
encoder.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
encoder.maxpool=nn.Identity()
encoder.fc=nn.Identity()

state_dict = torch.load("models/simclr_encoder.pt", map_location=torch.device('cpu'))
new_state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
encoder.load_state_dict(new_state_dict)
new_state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
encoder.load_state_dict(new_state_dict)
encoder.to(device)


for param in encoder.parameters():
    param.requires_grad = False
classifier=nn.Linear(512,10)
classifier.to(device)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(classifier.parameters(),lr=3e-4)
num_epochs=20
best_val_acc=0.0
for epoch in range(num_epochs):
    classifier.train()
    total_loss=0.0
    correct=0
    total=0
   
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            features = encoder(images)  
        outputs = classifier(features)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() 
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_acc =100*correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
    classifier.eval()
    correct=0
    total=0
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            features = encoder(images)  
            outputs = classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total
    print(f'Validation Acc: {val_acc:.2f}%')
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(classifier.state_dict(), "models/best_classifier.pt")
classifier.load_state_dict(torch.load("models/best_classifier.pt"))
classifier.eval()
correct=0
total=0
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        features = encoder(images)  
        outputs = classifier(features)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
test_acc = 100 * correct / total
print(f'Test Acc: {test_acc:.2f}%')
plt.figure(figsize=(10, 6))
plt.plot(train_accuracies, label='Train Accuracy', linewidth=2)
plt.plot(val_accuracies, label='Validation Accuracy', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Linear Probing: SimCLR Frozen Encoder', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graphs/linear_probe_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Linear probe accuracy plot saved!")
torch.save(classifier.state_dict(), "models/linear_probe.pt")
print(f"\n{'='*50}")
print(f"LINEAR PROBE RESULTS")
print(f"{'='*50}")
print(f"Random Encoder Linear Probe  Test Acc: {random_test_acc:.2f}%")
print(f"SimCLR Encoder Linear Probe  Test Acc: {test_acc:.2f}%")
print(f"{'='*50}")
print("✓ Linear probe classifier saved!")
# MSDS25008_05_task6_finetuning.py
import os
import torch
import random
import numpy as np
import torchvision
import torchvision.transforms as  T
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
# Set random seeds for reproducibility
SEED=2026
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
# Check for GPU availability
device=torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device} ')
# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('graphs', exist_ok=True)
# Function to load indices from text files
def load_split_indices(file_path):
    with open(file_path,'r') as f:
        indices = [int(line.strip()) for line in f]
    return indices
transform = T.Compose ([
    T.ToTensor(),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

])
# Load datasets and create dataloaders for the specified splits
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
val_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_indices = load_split_indices("splits/train_labeled_10percent.txt")
val_indices   = load_split_indices("splits/val.txt")
test_indices  = load_split_indices("splits/test.txt")
train_loader= DataLoader(Subset(train_dataset, train_indices),batch_size=64,shuffle=True)
val_loader  = DataLoader(Subset(val_dataset, val_indices), batch_size=64, shuffle=False)
test_loader = DataLoader(Subset(test_dataset, test_indices), batch_size=64, shuffle=False)
encoder=torchvision.models.resnet18(weights=None)
encoder.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
encoder.maxpool = nn.Identity()
encoder.fc      = nn.Identity()  # outputs 512-d features
state_dict = torch.load("models/simclr_encoder.pt", map_location='cpu')
new_state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
encoder.load_state_dict(new_state_dict)
print("✓ SimCLR encoder weights loaded")
model=nn.Sequential(encoder,nn.Linear(512,10)).to(device)
# Unfreeze all layers for fine-tuning
for param in encoder.parameters():
    param.requires_grad = True
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
# Fine-tuning loop with early stopping
num_epochs = 20
best_val_acc = 0.0
train_accuracies=[]
val_accuracies=[]
patience = 3
patience_counter = 0
for epoch in range(num_epochs):
    model.train()
    correct, total, total_loss = 0, 0,0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_acc = 100*correct / total
    train_accuracies.append(train_acc)
# model evaluation on validation set
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images,labels in val_loader:
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            _,predicted=torch.max(outputs.data,1)
            val_total+=labels.size(0)
            val_correct+=(predicted == labels).sum().item()
    val_acc=100*val_correct/val_total
    val_accuracies.append(val_acc)
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Loss: {total_loss/len(train_loader):.4f} "
          f"Train Acc: {train_acc:.2f}% "
          f"Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
         best_val_acc = val_acc
         patience_counter = 0

         torch.save(model.state_dict(), "models/finetuned_model.pt")
         print("  → Best model saved!")

    else:
        patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
model.load_state_dict(torch.load("models/finetuned_model.pt"))
model.eval()
correct=0
total=0
# Final evaluation on test set
with torch.no_grad():
     for images, labels in tqdm (test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

finetune_test_acc = 100 * correct / total
print(f"\nFine-tuned Model Test Accuracy: {finetune_test_acc:.2f}%")

# ── Accuracy plot 
plt.figure(figsize=(10, 6))
plt.plot(train_accuracies, label='Train Accuracy', linewidth=2)
plt.plot(val_accuracies,   label='Validation Accuracy', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Fine-tuning: SimCLR Pretrained Encoder', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graphs/finetuning_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Fine-tuning accuracy plot saved!")

#Final comparison
print(f"\n{'='*55}")
print(f"FINAL RESULTS COMPARISON")
print(f"{'='*55}")
print(f"{'Experiment':<40} {'Test Acc':>10}")
print(f"{'-'*55}")
print(f"{'SimCLR Encoder + Full Fine-tuning':<40} {finetune_test_acc:>9.2f}%")
print(f"{'='*55}")
print(f"\nNote: Add supervised, random probe, and SimCLR probe")
print(f"accuracies from earlier tasks to complete the table.")
# MSDS25008_05_task8_pca_tsne.py
import os
import json
import torch
import random
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
SEED = 2026
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

os.makedirs('results', exist_ok=True)
# Function to load indices from text files
def load_split_indices(file_path):
    with open(file_path, 'r') as f:
        return [int(line.strip()) for line in f]
# Data preparation
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616))
])

val_dataset  = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

val_indices  = load_split_indices("splits/val.txt")
test_indices = load_split_indices("splits/test.txt")

# Fixed 1000 validation images with seed 2026
rng = np.random.RandomState(SEED)
selected = rng.choice(len(val_indices), size=1000, replace=False)
subset_indices = [val_indices[i] for i in selected]

val_loader  = DataLoader(Subset(val_dataset,  subset_indices), batch_size=128, shuffle=False)
test_loader = DataLoader(Subset(test_dataset, test_indices),   batch_size=64,  shuffle=False)
# Class names for visualization
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Build encoder architecture
def build_encoder():
    enc = torchvision.models.resnet18(weights=None)
    enc.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    enc.maxpool = nn.Identity()
    enc.fc      = nn.Identity()
    return enc

# Extract features using the encoder
def extract_features(encoder, loader, device):
    encoder.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = encoder(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_features), np.concatenate(all_labels)

# Visualization function 
def plot_embeddings(embeddings, labels, title, save_path):
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab10', 10)
    for cls in range(10):
        mask = labels == cls
        plt.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=[colors(cls)],
            label=class_names[cls],
            alpha=0.6,
            s=15
        )
    plt.title(title, fontsize=14)
    plt.legend(fontsize=9, markerscale=2, loc='best')
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")
# function to load SimCLR weights into the encoder

def load_simclr_weights(encoder):
    state_dict = torch.load("models/simclr_encoder.pt", map_location='cpu')
    new_state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
    encoder.load_state_dict(new_state_dict)
    return encoder

print("\n── Random Encoder ──")
random_encoder = build_encoder().to(device)
random_features, labels = extract_features(random_encoder, val_loader, device)

tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
random_tsne = tsne.fit_transform(random_features)
plot_embeddings(random_tsne, labels,
                title="t-SNE: Random Untrained Encoder",
                save_path="results/random_encoder_pca_or_tsne.png")

print("\n── SimCLR Encoder ──")
simclr_encoder = build_encoder().to(device)
simclr_encoder = load_simclr_weights(simclr_encoder)
print("✓ SimCLR weights loaded")

simclr_features, _ = extract_features(simclr_encoder, val_loader, device)

tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
simclr_tsne = tsne.fit_transform(simclr_features)
plot_embeddings(simclr_tsne, labels,
                title="t-SNE: SimCLR Pretrained Encoder",
                save_path="results/simclr_encoder_pca_or_tsne.png")


print("\n── Fine-tuned Encoder ──")
finetuned_full = nn.Sequential(build_encoder(), nn.Linear(512, 10))
finetuned_full.load_state_dict(
    torch.load("models/finetuned_model.pt", map_location='cpu'))
finetuned_full = finetuned_full.to(device)
finetuned_encoder = finetuned_full[0]

finetuned_features, _ = extract_features(finetuned_encoder, val_loader, device)

tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
finetuned_tsne = tsne.fit_transform(finetuned_features)
plot_embeddings(finetuned_tsne, labels,
                title="t-SNE: Fine-tuned Encoder",
                save_path="results/finetuned_encoder_pca_or_tsne.png")

print("\n── Generating metrics.json ──")

# Final metrics dictionary
metrics ={
    "student_name": "Neeka Javed",
    "roll_number": "MSDS25008",
    "seed": 2026,
    "batch_size": 64,
    "simclr_epochs": 50,
    "linear_probe_epochs": 20,
    "finetuning_epochs": 20,
    "learning_rate": 0.0003,
    "temperature": 0.5,
    "supervised_10percent_test_acc": 68.46,
    "random_linear_probe_test_acc": 27.56,
    "simclr_linear_probe_test_acc": 74.40,
    "simclr_finetune_test_acc": 81.20,
    "same_view_similarity_before": 0.8897,
    "different_image_similarity_before": 0.8767,
    "same_view_similarity_after": 0.9121,
    "different_image_similarity_after": 0.3475
}

with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("✓ Saved: results/metrics.json")
# Generate test_predictions.csv

print("\n── Generating test_predictions.csv ──")

finetuned_full.eval()
all_image_indices = []
all_true_labels   = []
all_pred_labels   = []
all_probs         = []

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        outputs = finetuned_full(images)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        _, predicted = torch.max(outputs, 1)

        start_idx = batch_idx * test_loader.batch_size
        for i in range(len(labels)):
            all_image_indices.append(test_indices[start_idx + i])
            all_true_labels.append(labels[i].item())
            all_pred_labels.append(predicted[i].item())
            all_probs.append(probs[i])

# Build dataframe
probs_array = np.array(all_probs)
df = pd.DataFrame({
    "image_index":     all_image_indices,
    "true_label":      all_true_labels,
    "predicted_label": all_pred_labels,
})
for c in range(10):
    df[f"prob_class_{c}"] = probs_array[:, c]

df.to_csv("results/test_predictions.csv", index=False)
print("✓ Saved: results/test_predictions.csv")
print(f"\n{'='*50}")
print("ALL OUTPUTS GENERATED")
print(f"{'='*50}")
print("results/random_encoder_pca_or_tsne.png")
print("results/simclr_encoder_pca_or_tsne.png")
print("results/finetuned_encoder_pca_or_tsne.png")
print("results/metrics.json")
print("results/test_predictions.csv")




