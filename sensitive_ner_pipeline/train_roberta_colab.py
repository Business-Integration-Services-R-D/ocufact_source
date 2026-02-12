"""
train_roberta_colab.py

Google Colab version of train_roberta.py with:
- Google Drive mounting
- A100 GPU setup and verification
- Global variables for easy path configuration

Setup Instructions:
1. In Colab: Runtime > Change runtime type > Hardware accelerator > GPU (A100)
2. Run this script
3. It will mount your Google Drive and save models there
"""

# ========================
# COLAB SETUP
# ========================
import os

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Verify GPU
import torch
print("="*80)
print("GPU SETUP VERIFICATION")
print("="*80)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    # Check if A100
    gpu_name = torch.cuda.get_device_name(0)
    if "A100" in gpu_name:
        print("✅ A100 GPU detected!")
    else:
        print(f"⚠️  Warning: Not an A100 GPU. Current: {gpu_name}")
else:
    print("❌ No GPU available! Please enable GPU in Runtime > Change runtime type")
print("="*80)
print()

# Install required packages (if not already installed)
print("Installing/Upgrading required packages...")
os.system("pip install -q transformers datasets scikit-learn accelerate")
print("✅ Packages installed\n")

# ========================
# GLOBAL CONFIGURATION
# ========================

# Google Drive paths - MODIFY THESE to match your Drive structure
DRIVE_BASE = "/content/drive/MyDrive/sensitive_data_detection"  # Base project folder in Drive
DATASET_PATH = os.path.join(DRIVE_BASE, "merged_ner_dataset")
OUT_DIR = os.path.join(DRIVE_BASE, "merged_ner_models", "roberta-large")

# Model configuration
MODEL_CHECKPOINT = "FacebookAI/roberta-large"

# Training hyperparameters
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 4  # Increased for A100 (can go higher if you have memory)
EVAL_BATCH_SIZE = 4
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIMIT = 2

# Mixed precision training (recommended for A100)
USE_FP16 = True  # A100 supports FP16, set to False if issues occur
USE_BF16 = False  # A100 also supports BF16, can experiment with this

# Gradient accumulation (if you want effective larger batch size)
GRADIENT_ACCUMULATION_STEPS = 1

print("="*80)
print("CONFIGURATION")
print("="*80)
print(f"Dataset path: {DATASET_PATH}")
print(f"Output directory: {OUT_DIR}")
print(f"Model: {MODEL_CHECKPOINT}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Batch size (train/eval): {TRAIN_BATCH_SIZE}/{EVAL_BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"FP16: {USE_FP16}, BF16: {USE_BF16}")
print("="*80)
print()

# ========================
# IMPORTS
# ========================
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ========================
# LOAD DATASET
# ========================
print(f"Loading dataset from: {DATASET_PATH}")
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {DATASET_PATH}. "
        f"Please ensure the dataset is in your Google Drive at this location."
    )

dataset = load_from_disk(DATASET_PATH)
print(f"✅ Dataset loaded successfully")
print(f"   - Train samples: {len(dataset['train'])}")
print(f"   - Validation samples: {len(dataset['validation'])}")
if "test" in dataset:
    print(f"   - Test samples: {len(dataset['test'])}")
print()

# Extract label information
label_list = dataset["train"].features["ner_tags"].feature.names
label_list = list(label_list)
num_labels = len(label_list)
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

print(f"Number of labels: {num_labels}")
print(f"Labels: {label_list[:10]}..." if len(label_list) > 10 else f"Labels: {label_list}")
print()

# ========================
# TOKENIZER
# ========================
print(f"Loading tokenizer: {MODEL_CHECKPOINT}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, add_prefix_space=True)
print("✅ Tokenizer loaded\n")

# ========================
# TOKENIZATION FUNCTION
# ========================
def tokenize_and_align_labels(examples):
    """
    Tokenize inputs and align labels with subword tokens.
    Subword tokens (continuation tokens) are assigned -100 to ignore in loss.
    """
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100
                label_ids.append(-100)
            elif word_idx != previous_idx:
                # First subword token of a word gets the label
                label_ids.append(label[word_idx])
            else:
                # Continuation subword tokens get -100
                label_ids.append(-100)
            previous_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

print("Tokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
print("✅ Dataset tokenized\n")

# ========================
# MODEL
# ========================
print(f"Loading model: {MODEL_CHECKPOINT}")
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)
print("✅ Model loaded")
print(f"   - Parameters: {model.num_parameters():,}")
print()

# ========================
# METRICS FUNCTION
# ========================
def compute_metrics(p):
    """
    Compute precision, recall, F1, and accuracy for NER task.
    Ignores -100 labels (padding/special tokens).
    """
    logits, labels = p
    preds = logits.argmax(axis=-1)
    
    # Convert IDs back to label names, excluding -100
    true_labels = [[label_list[l] for l in seq if l != -100] for seq in labels]
    true_preds = [[label_list[p] for (p, l) in zip(pr, seq) if l != -100]
                  for pr, seq in zip(preds, labels)]
    
    # Flatten for sklearn metrics
    flat_true = [l for seq in true_labels for l in seq]
    flat_pred = [p for seq in true_preds for p in seq]
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_true, flat_pred, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(flat_true, flat_pred)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }

# ========================
# TRAINING ARGUMENTS
# ========================
print("Setting up training arguments...")
os.makedirs(OUT_DIR, exist_ok=True)

args = TrainingArguments(
    OUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    save_total_limit=SAVE_TOTAL_LIMIT,
    logging_dir=os.path.join(OUT_DIR, "logs"),
    logging_steps=50,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    fp16=USE_FP16,
    bf16=USE_BF16,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",  # Disable wandb/tensorboard for Colab simplicity
    push_to_hub=False,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

# ========================
# TRAINER
# ========================
print("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("✅ Trainer initialized\n")

# ========================
# TRAINING
# ========================
print("="*80)
print("STARTING TRAINING")
print("="*80)
print()

trainer.train()

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print()

# ========================
# SAVE MODEL
# ========================
print(f"Saving model to: {OUT_DIR}")
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print(f"✅ Model and tokenizer saved to {OUT_DIR}")
print()

# ========================
# FINAL EVALUATION
# ========================
print("="*80)
print("FINAL EVALUATION")
print("="*80)
eval_results = trainer.evaluate()
print("\nValidation Results:")
for key, value in eval_results.items():
    print(f"  {key}: {value:.4f}")
print()

# If test set exists, evaluate on it
if "test" in tokenized_datasets:
    print("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_datasets["test"])
    print("\nTest Results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")
    print()

print("="*80)
print("✅ ALL DONE!")
print("="*80)
print(f"\nYour trained model is saved in Google Drive at:")
print(f"  {OUT_DIR}")
print("\nYou can download it or use it for inference directly from Colab.")


