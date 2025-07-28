import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score
from collections import defaultdict
import matplotlib.pyplot as plt
from CNN.MixtureMasking import MixtureMasking
from CNN_final.model import ViolinViolaCNN, ViolinViolaCrossEntropyLoss, ViolinViolaDataset, load_model

transform = transforms.Compose([
    transforms.Resize((128, 517)),  # Resize to a fixed size
    transforms.Normalize([0.5], [0.5])  # Normalize the image
])

g_test = torch.Generator().manual_seed(17)
dataset = ViolinViolaDataset('CNN/test/', transform=transform)
#g_train = torch.Generator().manual_seed(17)
#train_dataset = ViolinViolaDataset('CNN/train/', transform=transform)
#combined_dataset = torch.utils.data.ConcatDataset([train_dataset, dataset])
single_sample_dataset = Subset(dataset, list(range(32)))
loader = DataLoader(single_sample_dataset, batch_size=32, shuffle=False)
#loader = DataLoader(combined_dataset, batch_size=32, shuffle=False)
    
model = load_model()
model.eval()

# Loss function
criterion = ViolinViolaCrossEntropyLoss()

# Accumulators
total_loss = 0.0
total_correct = 0
total_samples = 0
class_correct = defaultdict(int)
class_total = defaultdict(int)

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for inputs, labels in loader:
        inputs, labels = inputs.to('cpu'), torch.tensor(labels).to('cpu')

        outputs = model(inputs, labels)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)

        # Predictions
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        # Store for metrics
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs[:, 1].tolist())  # Prob for class 1

        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        for l, p in zip(labels, preds):
            class_total[l.item()] += 1
            if l == p:
                class_correct[l.item()] += 1

# Metrics
avg_loss = total_loss / total_samples
accuracy = total_correct / total_samples
f1 = f1_score(all_labels, all_preds, average='macro')
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')

print(f"Total Samples: {total_samples}")
print(f"Loss: {avg_loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
for cls in sorted(class_total.keys()):
    acc = class_correct[cls] / class_total[cls]
    print(f"Class {cls} Accuracy: {acc:.4f}")
print(f"F1 Score (macro): {f1:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random baseline')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()