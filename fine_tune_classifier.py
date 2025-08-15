from datasets import load_dataset
from transformers import AutoImageProcessor, Dinov2ForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from torchvision.transforms import Compose, ToTensor, Normalize, RandomResizedCrop
import torch
from sklearn.metrics import f1_score
import numpy as np

# Load dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--label_type', required=True, help='Specify the label type')
args = parser.parse_args()

label_type = args.label_type
data_dir = f"{label_type}/ds"
dataset = load_dataset("imagefolder", data_dir=data_dir)

# Process labels
labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()

for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

# Define checkpoint
CHECKPOINT = "facebook/dinov2-large"

# Load image processor
image_processor = AutoImageProcessor.from_pretrained(CHECKPOINT)

# Define image size
SIZE = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)

# Define transforms
_transforms = Compose([
    RandomResizedCrop(SIZE, antialias=True),
    ToTensor(),
    # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

# Apply transforms to dataset
dataset = dataset.with_transform(transforms)

# Define collate function
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# Load model
model = Dinov2ForImageClassification.from_pretrained(
    CHECKPOINT,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

model.to("cuda")

def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=1)
    macro_f1 = f1_score(labels, predictions, average="macro")
    return {"macro_f1": macro_f1}

# Training arguments
STRATEGY = "epoch"
OUTPUT_DIR = f"{label_type}/dinov2"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy=STRATEGY,
    save_strategy=STRATEGY,
    save_total_limit=2,
    logging_steps=10,

    remove_unused_columns=False,

    learning_rate=1e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=8,
    num_train_epochs=25,
    warmup_ratio=0.1,

    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",  # Optimize for macro F1 score
    greater_is_better=True,  # Higher F1 is better
    push_to_hub=False,
    report_to="none"
)

# # Early stopping callback
# early_stopping_callback = EarlyStoppingCallback(
#     early_stopping_patience=50,  # Patience of 50 epochs
# )

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    # callbacks=[early_stopping_callback],  # Add early stopping here if needed
)

# Train the model
trainer.train()
