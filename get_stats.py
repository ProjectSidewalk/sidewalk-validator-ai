from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import random
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix, balanced_accuracy_score, precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import json

# Load dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--label_type', required=True, help='Specify the label type')
args = parser.parse_args()

label_type = args.label_type
data_dir = f"{label_type}/ds"
dataset = load_dataset("imagefolder", data_dir=data_dir)

# Define checkpoint directory and device
CHECKPOINT_BASE_DIR = f"{label_type}/dinov2"

# Determine the latest checkpoint directory
checkpoint_dirs = [
    d for d in os.listdir(CHECKPOINT_BASE_DIR)
    if os.path.isdir(os.path.join(CHECKPOINT_BASE_DIR, d)) and d.startswith("checkpoint-")
]
checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]), reverse=True)

if checkpoint_dirs:
    latest_checkpoint_dir = os.path.join(CHECKPOINT_BASE_DIR, checkpoint_dirs[0])
    trainer_state_path = os.path.join(latest_checkpoint_dir, "trainer_state.json")

    # Read the best model checkpoint from trainer_state.json
    with open(trainer_state_path, "r") as f:
        trainer_state = json.load(f)
    best_model_checkpoint = trainer_state.get("best_model_checkpoint")

    if best_model_checkpoint:
        CHECKPOINT_DIR = best_model_checkpoint
    else:
        raise ValueError("best_model_checkpoint not found in trainer_state.json.")
else:
    raise FileNotFoundError("No checkpoint directories found.")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load image processor and model
image_processor = AutoImageProcessor.from_pretrained(CHECKPOINT_DIR)
model = AutoModelForImageClassification.from_pretrained(CHECKPOINT_DIR)
model = model.to(DEVICE)

# Define true labels and predicted logits
true_labels = []
predicted_logits = []

# Loop through test set for evaluation
for sample in tqdm(dataset["test"]):
    image = sample["image"]
    label = sample["label"]
    true_labels.append(label)

    inputs = image_processor(image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_logits.append(logits.cpu())

# Convert logits to predictions
predicted_probs = torch.cat(predicted_logits).softmax(dim=-1).numpy()
predicted_labels = predicted_probs.argmax(axis=-1)

# Evaluate metrics
accuracy = balanced_accuracy_score(true_labels, predicted_labels)
report = classification_report(true_labels, predicted_labels, target_names=model.config.id2label.values())
conf_matrix = confusion_matrix(true_labels, predicted_labels)

print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy: {accuracy:.4f}")

# Precision vs Recall Curve
for class_id, class_name in model.config.id2label.items():
    precision, recall, _ = precision_recall_curve(
        [1 if label == class_id else 0 for label in true_labels],
        predicted_probs[:, class_id],
    )
    plt.plot(recall, precision, label=f"{class_name}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision vs Recall Curve")
plt.legend(loc="lower left")
plt.savefig(f"{label_type}/pvr.png")
plt.show()

# Import necessary libraries for text rendering
from matplotlib.gridspec import GridSpec

# Prepare a figure for the combined report
fig = plt.figure(figsize=(10, 15))
gs = GridSpec(2, 1, height_ratios=[1, 2], figure=fig)  # 1:2 ratio for graph and report

# Plot Precision vs Recall Curve
ax_pr = fig.add_subplot(gs[0])
for class_id, class_name in model.config.id2label.items():
    precision, recall, thresholds = precision_recall_curve(
        [1 if label == class_id else 0 for label in true_labels],
        predicted_probs[:, class_id],
    )
    ax_pr.plot(recall, precision, label=f"{class_name}")

# Add horizontal lines for 90% and 95% precision
ax_pr.axhline(y=0.9, color='red', linestyle='--', label="90% Precision")
ax_pr.axhline(y=0.95, color='blue', linestyle='--', label="95% Precision")

ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.set_title(f"Precision vs Recall Curve ({label_type})")
ax_pr.legend(loc="lower left")
ax_pr.grid(True)

# Render Classification Report below the graph
ax_report = fig.add_subplot(gs[1])
ax_report.axis("off")  # Hide axes for text rendering

# Define precision levels for which we want to find recall
precision_levels = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

recallAtVariousPrecisionLevelsText = ""

# Loop through each class to calculate recall at specified precision levels
for class_id, class_name in model.config.id2label.items():
    precision, recall, thresholds = precision_recall_curve(
        [1 if label == class_id else 0 for label in true_labels],
        predicted_probs[:, class_id],
    )
    recallAtVariousPrecisionLevelsText += f"Recall at precision levels for class '{class_name}':\n"
    print(f"Recall at precision levels for class '{class_name}':\n")
    for precision_level in precision_levels:
        # Find the threshold for the given precision level
        precision_idx = np.searchsorted(precision, precision_level)
        if precision_idx < len(precision):
            corresponding_recall = recall[precision_idx]
            print(f"  Precision {precision_level*100:.0f}%: Recall = {corresponding_recall:.4f}")
            recallAtVariousPrecisionLevelsText += f"  Precision {precision_level*100:.0f}%: Recall = {corresponding_recall:.4f}\n"
        else:
            print(f"  Precision {precision_level*100:.0f}%: Recall = N/A (no value found)")

# Convert classification report to string and render as text
classification_report_text = classification_report(
    true_labels,
    predicted_labels,
    target_names=model.config.id2label.values()
)
ax_report.text(
    0.01, 0.5, classification_report_text+"\n"+recallAtVariousPrecisionLevelsText,
    fontsize=10,
    va="center",
    ha="left",
    family="monospace",
    transform=ax_report.transAxes
)

# Save the combined report to an image file
plt.tight_layout()
plt.savefig(f"{label_type}/report.png")
plt.show()

confidence_scores = predicted_probs.max(axis=-1)
confidence_bins = np.linspace(0.5, 1, 20001)  # Bins from 0.5 to 1.0
classes = np.unique(true_labels)  # Unique class labels

# Initialize a dictionary to store precision scores for each class
class_precisions = {cls: np.zeros(len(confidence_bins) - 1) for cls in classes}

# Iterate through bins
for bin_id in range(1, len(confidence_bins)):
    threshold = confidence_bins[bin_id - 1]
    
    # Mask where confidence scores are greater than or equal to the bin threshold
    bin_mask = confidence_scores >= threshold
    
    # Compute precision for each class
    for cls in classes:
        if bin_mask.sum() > 0:  # Ensure there are samples in this bin
            bin_precision = precision_score(
                [true_labels[i] == cls for i in range(len(true_labels)) if bin_mask[i]],
                [predicted_labels[i] == cls for i in range(len(predicted_labels)) if bin_mask[i]],
                zero_division=0,  # Handle cases with no positive predictions
            )
            class_precisions[cls][bin_id - 1] = bin_precision
        else:
            class_precisions[cls][bin_id - 1] = 0.0

# Plot Precision vs Confidence for each class
# Plot Precision vs Confidence for each class
plt.figure()
for cls in classes:
    class_name = model.config.id2label[cls]  # Get the human-readable class name
    plt.plot(
        confidence_bins[:-1],
        class_precisions[cls],
        marker="o",
        label=f"{class_name}"  # Use the class name in the label
    )
plt.xlabel("Confidence")
plt.ylabel("Precision")
plt.title("Precision vs Confidence (Cumulative) by Class")
plt.grid(True)
plt.legend()
plt.savefig(f"{label_type}/pvc_cumulative_by_class.png")
plt.show()

# Create a dictionary to store confidence-to-precision mappings for each class
confidence_precision_map = {}

# Iterate through each class
for class_id in classes:
    class_name = model.config.id2label[class_id]  # Get the human-readable class name
    class_confidence_precisions = {}  # Store confidence-to-precision for this class

    prev_precision = 0
    
    # Store the precision values for each confidence bin
    for bin_id in range(1, len(confidence_bins)):
        threshold = confidence_bins[bin_id - 1]
        this_precision = class_precisions[class_id][bin_id - 1]
        if this_precision != prev_precision:
            class_confidence_precisions[threshold] = class_precisions[class_id][bin_id - 1]
            prev_precision = this_precision
    
    # Add the dictionary for this class to the main map
    confidence_precision_map[class_name] = class_confidence_precisions

# Save the dictionary to a JSON file
json_output_path = f"{label_type}/confidence_precision_map.json"
with open(json_output_path, 'w') as f:
    json.dump(confidence_precision_map, f, indent=4)

print(f"JSON file saved at {json_output_path}")

# Define the desired precision level
desired_precision = 0.95

# Loop through each class to find the threshold for the desired precision
for class_id, class_name in model.config.id2label.items():
    precision, recall, thresholds = precision_recall_curve(
        [1 if label == class_id else 0 for label in true_labels],
        predicted_probs[:, class_id],
    )
    
    # Find the index of the threshold where precision is closest to desired_precision
    precision_idx = np.searchsorted(precision, desired_precision, side='left')

    if precision_idx < len(precision) and precision[precision_idx] >= desired_precision:
        threshold = thresholds[precision_idx]
        print(f"Class '{class_name}': {desired_precision} Precision corresponds to threshold = {threshold:.4f}")
    else:
        print(f"Class '{class_name}': {desired_precision} Precision is not achievable (no suitable threshold found)")