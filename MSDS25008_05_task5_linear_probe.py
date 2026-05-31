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
