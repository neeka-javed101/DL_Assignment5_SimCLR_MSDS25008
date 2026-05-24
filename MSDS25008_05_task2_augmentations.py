import os
import random
import torch
import numpy as np
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
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

raw_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True, 
    transform=None  
)
def tensor_to_image(tensor):
    """Convert normalized tensor [C, H, W] to numpy image [H, W, C]"""
    # ✅ Denormalize (reverse the Normalize transform)
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
