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
num_epochs = 30
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
best_val_accuracy = 0
patience = 3
patience_counter = 0

    
    
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


# ---------------- CONFUSION MATRIX ---------------- #

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