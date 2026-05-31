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
