from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Paths & Model
DATASET_PATH = os.path.join("..", "merged_ner_dataset")
model_checkpoint = "FacebookAI/roberta-large"
OUT_DIR = os.path.join("..", "merged_ner_models", "roberta-large")

# Load dataset
dataset = load_from_disk(DATASET_PATH)
label_list = dataset["train"].features["ner_tags"].feature.names
label_list = list(label_list)
num_labels = len(label_list)
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

# Tokenizer with add_prefix_space=True for RoBERTa
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

# Tokenization helper
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Model
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

# Metrics function
def compute_metrics(p):
    logits, labels = p
    preds = logits.argmax(axis=-1)
    true_labels = [[label_list[l] for l in seq if l != -100] for seq in labels]
    true_preds = [[label_list[p] for (p, l) in zip(pr, seq) if l != -100]
                  for pr, seq in zip(preds, labels)]
    flat_true = [l for seq in true_labels for l in seq]
    flat_pred = [p for seq in true_preds for p in seq]
    precision, recall, f1, _ = precision_recall_fscore_support(flat_true, flat_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(flat_true, flat_pred)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }

# Training arguments
args = TrainingArguments(
    OUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir=os.path.join(OUT_DIR, "logs"),
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print(f"✅ Model saved to {OUT_DIR}")
