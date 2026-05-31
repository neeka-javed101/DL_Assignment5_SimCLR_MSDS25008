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


