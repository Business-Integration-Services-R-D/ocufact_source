from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Paths
DATASET_PATH = "binary_class_dataset"
model_checkpoint = "FacebookAI/roberta-large"
OUT_DIR = "models/" + "roberta-large"
# Load dataset
dataset = load_from_disk(DATASET_PATH)
label_list = dataset["train"].features["ner_tags"].feature.names
label_list = list(label_list)

id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}
num_labels = len(label_list)

# Model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Tokenization
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
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

# Custom seqeval-like metrics function
def compute_seqeval_metrics(true_labels, true_predictions):
    """
    Compute precision, recall, F1, and accuracy for sequence labeling
    Similar to seqeval but using sklearn metrics
    """
    # Flatten the lists for token-level evaluation
    flat_true = []
    flat_pred = []
    
    for true_seq, pred_seq in zip(true_labels, true_predictions):
        flat_true.extend(true_seq)
        flat_pred.extend(pred_seq)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_true, flat_pred, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(flat_true, flat_pred)
    
    return {
        "overall_precision": precision,
        "overall_recall": recall,
        "overall_f1": f1,
        "overall_accuracy": accuracy
    }

# Metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)

    true_labels = [
        [label_list[l] for l in label if l != -100]
        for label in labels
    ]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = compute_seqeval_metrics(true_labels, true_predictions)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}

# Training
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
    logging_dir=f"{OUT_DIR}/logs",
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

# Save final model
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print(f"✅ Model saved to {OUT_DIR}")
