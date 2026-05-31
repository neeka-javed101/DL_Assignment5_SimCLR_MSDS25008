
    # Deep Learning Assignment 5: SimCLR Implementation
## Self-Supervised Learning from Scratch

**Student:** Neeka Javeed  
**Roll Number:** MSDS25008  
**Course:** Deep Learning - Spring 2026  
**Assignment:** SimCLR Self-Supervised Learning  
**Random Seed:** 2026

---

## 📋 Project Overview

This project implements **SimCLR (Simple Framework for Contrastive Learning of Visual Representations)**, a self-supervised learning method that learns useful image representations without class labels.

### Key Idea
SimCLR learns by:
1. Creating two augmented views of the same image (positive pair)
2. Pushing similar representations together
3. Pushing different images' representations apart
4. Using NT-Xent contrastive loss

---

## 🏗️ Project Structure
MSDS25008_Assignment5/
├── splits/
│   ├── train_ssl_unlabeled.txt      (45k unlabeled images for pretraining)
│   ├── train_labeled_10percent.txt  (5k labeled images - 10% of training)
│   ├── val.txt                       (5k validation images)
│   └── test.txt                      (10k test images)
│
├── data/
│   └── cifar-10/                     (Auto-downloaded on first run)
│
├── models/
│   ├── supervised_best.pt            (Supervised baseline - Task 1)
│   ├── simclr_encoder.pt             (SimCLR pretrained encoder - Task 4)
│   ├── linear_probe.pt               (Linear classifier - Task 5)
│   └── finetuned_model.pt            (Fine-tuned model - Task 6)
│
├── graphs/
│   ├── supervised_loss.png           (Task 1)
│   ├── simclr_pretraining_loss.png   (Task 5)
│   ├── linear_probe_accuracy.png     (Task 5)
│   └── finetuning_accuracy.png       (Task 6)
│
├── results/
│   ├── supervised_confusion_matrix.png
│   ├── augmentation_examples.png
│   ├── similarity_matrix_before_training.png
│   ├── similarity_matrix_after_training.png
│   ├── random_encoder_pca_or_tsne.png
│   ├── simclr_encoder_pca_or_tsne.png
│   ├── finetuned_encoder_pca_or_tsne.png
│   ├── finetuned_confusion_matrix.png
│   ├── metrics.json
│   └── test_predictions.csv
│
├── MSDS25008_05_task1_supervised.py
├── MSDS25008_05_task2_augmentations.py
├── MSDS25008_05_task3_similarity.py
├── MSDS25008_05_task4_simclr.py
├── MSDS25008_05_task5_linear_probe.py
├── MSDS25008_05_task6_finetune.py
├── MSDS25008_05_task8_pca_tsne.py
├── MSDS25008_05_allCode.py           (Combined code)
│
├── requirements.txt
├── README.md
└── Report.pdf                        (Final report)

---

## 📅 Checkpoint Breakdown

### **Checkpoint 1 - Day 3** ✅
- Fixed dataset split loading
- Supervised baseline training (10% labels)
- Augmentation pipeline
- Two-view transform
- Augmentation visualization
- **Output:** `supervised_loss.png`, `supervised_confusion_matrix.png`

### **Checkpoint 2 - Day 6** ✅
- Encoder implementation (ResNet-18 modified)
- Projection head implementation
- Positive and negative pair construction
- Similarity matrix computation
- NT-Xent loss implementation
- **Output:** `similarity_matrix_before_training.png`

### **Checkpoint 3 - Day 9** ✅
- SimCLR pretraining (50 epochs)
- Loss curve generation
- Feature similarity before vs after training
- **Output:** `simclr_pretraining_loss.png`, `similarity_matrix_after_training.png`

### **Checkpoint 4 - Day 12** ✅
- Linear probing evaluation
- Fine-tuning full model
- PCA/t-SNE visualization
- metrics.json generation
- test_predictions.csv generation
- **Output:** All visualizations, metrics, predictions

---

## 🚀 How to Run

### **Prerequisites:**
```bash
pip install -r requirements.txt
```

### **Run Individual Tasks:**

```bash
# Task 1: Supervised Baseline
python MSDS25008_05_task1_supervised.py

# Task 2: Augmentation Visualization
python MSDS25008_05_task2_augmentations.py

# Task 3: Feature Similarity Before Training
python MSDS25008_05_task3_similarity.py

# Task 4: SimCLR Training
python MSDS25008_05_task4_simclr.py

# Task 5: Linear Probing
python MSDS25008_05_task5_linear_probe.py

# Task 6: Fine-tuning
python MSDS25008_05_task6_finetune.py

# Task 8: PCA/t-SNE & Metrics
python MSDS25008_05_task8_pca_tsne.py
```

### **Or Run All at Once:**
```bash
python MSDS25008_05_allCode.py
```

---

## 📊 Expected Results

| Task | Model | Test Accuracy |
|------|-------|---------------|
| **1** | Supervised (10% labels) | ~68-72% |
| **5** | Random Encoder + Linear Probe | ~27-30% |
| **5** | SimCLR Encoder + Linear Probe | ~74-76% |
| **6** | SimCLR + Fine-tuning | ~81-83% |

**Key Finding:** SimCLR pretraining significantly improves downstream classification compared to random features!

---

## 🔧 Configuration

### **Model Architecture**
- **Encoder:** ResNet-18 (modified for CIFAR-10)
  - Conv1: 3×3, stride=1, padding=1 (no maxpool)
  - Output: 512-dimensional features
- **Projection Head:** 512 → 256 → 128 (used only during pretraining)
- **Classifier Head:** 512 → 10 (for downstream tasks)

### **Training Settings**
- **Dataset:** CIFAR-10
- **Random Seed:** 2026
- **Batch Size:** 64
- **Learning Rate:** 3e-4
- **Optimizer:** Adam
- **Temperature (τ):** 0.5
- **SimCLR Epochs:** 50
- **Linear Probe Epochs:** 20
- **Fine-tuning Epochs:** 20

### **Augmentations (SimCLR)**
- RandomResizedCrop(32, scale=(0.2, 1.0))
- RandomHorizontalFlip(p=0.5)
- ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
- RandomGrayscale(p=0.2)
- Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

---

## 📈 Key Results

### **Similarity Analysis**
| Metric | Before Training | After Training |
|--------|-----------------|-----------------|
| Same image views | 0.8897 | 0.9121 ↑ |
| Different images | 0.8767 | 0.3475 ↓ |

**Interpretation:** SimCLR successfully learned to:
- Keep similar views close (same image similarity increased)
- Push different images apart (different image similarity decreased)

### **Accuracy Improvement**
Random Features:      27.56%
↓
SimCLR Frozen:       74.40%  (+46.84%)
↓
SimCLR Fine-tuned:   81.20%  (+53.64%)

---

## 🎨 Visualizations

### **t-SNE Embeddings**
- **Random Encoder:** Classes completely mixed, no structure
- **SimCLR Encoder:** Clear class separation, learned meaningful features
- **Fine-tuned Encoder:** Even better separation, task-optimized

### **Loss Curves**
- **Supervised:** Convergence with early stopping
- **SimCLR:** Smooth decay over 50 epochs
- **Linear Probe:** Fast convergence with frozen encoder

---

## 📝 Implementation Details

### **NT-Xent Loss**
L(i,j) = -log[ exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ) ]
- Numerator: similarity of positive pair
- Denominator: all similarities (positive + negatives)
- Temperature τ controls sharpness

### **Positive Pair Construction**
- For batch of N images: create 2N augmented views
- Positive pairs: (i, i+N) and (i+N, i)
- All other pairs: negatives

### **Linear Probing Protocol**
- Freeze encoder weights
- Train only linear classifier head
- Evaluate on downstream task
- Shows quality of learned representations

### **Fine-tuning Protocol**
- Initialize with pretrained encoder
- Unfreeze all parameters
- Train end-to-end on downstream task
- Better performance with task-specific optimization

---

## 📦 Requirements
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
pandas>=1.1.0
seaborn>=0.11.0
tqdm>=4.50.0
pillow>=8.0.0

Install with:
```bash
pip install -r requirements.txt
```

---

## 🔄 Git Workflow

Regular commits were made according to checkpoint schedule:

```bash
# Checkpoint 1 (Day 3)
git commit -m "feat: Tasks 1-3 complete (Checkpoint 1)"

# Checkpoint 2 (Day 6)
git commit -m "feat: Tasks 4.1-4.4 complete (Checkpoint 2)"

# Checkpoint 3 (Day 9)
git commit -m "feat: Task 5 complete (Checkpoint 3)"

# Checkpoint 4 (Day 12)
git commit -m "feat: Tasks 6, 8 complete (Checkpoint 4)"
```

---

## 📚 References

### **Paper:**
Advancing Self-Supervised and Semi-Supervised Learning with SimCLR  
https://arxiv.org/abs/2002.05709

### **Key Concepts:**
- Self-supervised learning: Learning from data structure without labels
- Contrastive learning: Pushing similar samples together, dissimilar apart
- Data augmentation invariance: Augmentations preserve semantic content
- Representation learning: Learning features useful for downstream tasks
- Linear evaluation: Testing representation quality via linear probing
- Transfer learning: Using pretrained models for new tasks

---

## 🎓 Learning Outcomes

After completing this assignment, you should understand:
1. ✅ Difference between supervised and self-supervised learning
2. ✅ Why augmentations matter in SimCLR
3. ✅ How contrastive loss works
4. ✅ Positive and negative pair construction
5. ✅ Linear probing evaluation protocol
6. ✅ Fine-tuning vs frozen evaluation
7. ✅ Feature visualization and analysis

---

## ⚠️ Important Notes

1. **No Pre-trained Weights:** ResNet-18 is trained from scratch
2. **Fixed Splits:** Must use provided split files for reproducibility
3. **Seed:** Always use SEED=2026 for reproducibility
4. **Labels in Pretraining:** NOT used during SimCLR training
5. **Temperature:** τ=0.5 (fixed throughout experiments)
6. **Batch Size:** 64 (minimum 32 if GPU memory is limited)
7. **No External SimCLR Libraries:** Implementation from scratch only

---

## 🐛 Troubleshooting

### **GPU Memory Issues:**
Reduce batch size in code:
```python
batch_size = 32  # instead of 64
```

### **CIFAR-10 Download Issues:**
Ensure internet connection or manually download dataset

### **File Not Found Error:**
Ensure `splits/` folder exists in current directory with 4 `.txt` files

### **Model Loading Errors:**
Check encoder prefix removal when loading SimCLR weights

---

## 📋 File Descriptions

| File | Purpose |
|------|---------|
| `MSDS25008_05_task1_supervised.py` | Supervised baseline on 10% labels |
| `MSDS25008_05_task2_augmentations.py` | Augmentation visualization |
| `MSDS25008_05_task3_similarity.py` | Feature similarity analysis |
| `MSDS25008_05_task4_simclr.py` | SimCLR pretraining (50 epochs) |
| `MSDS25008_05_task5_linear_probe.py` | Linear probing evaluation |
| `MSDS25008_05_task6_finetune.py` | Fine-tuning on downstream task |
| `MSDS25008_05_task8_pca_tsne.py` | Visualization + metrics + predictions |
| `MSDS25008_05_allCode.py` | Combined all tasks |

---

## 📊 Metrics Generated

### **metrics.json Contains:**
- Student name and roll number
- All hyperparameters used
- Test accuracies for all 4 experiments
- Similarity metrics before/after training

### **test_predictions.csv Contains:**
- Image index
- True label
- Predicted label
- Probability for each class (0-9)

---

## 🏆 Key Achievements

✅ Implemented SimCLR from scratch without external libraries  
✅ Achieved 81.20% test accuracy with fine-tuning  
✅ Demonstrated 53.64% improvement over random features  
✅ Generated comprehensive visualizations (t-SNE, confusion matrices, loss curves)  
✅ Made regular checkpoint-based commits to GitHub  
✅ Followed assignment requirements strictly  

---

