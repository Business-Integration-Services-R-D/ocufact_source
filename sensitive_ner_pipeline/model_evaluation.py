from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, DataCollatorForTokenClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os

DATASET_PATH = r"binary_class_dataset"
MODEL_PATH = r"models\roberta-large"

# Load dataset + labels
dataset = load_from_disk(DATASET_PATH)
label_list = dataset["train"].features["ner_tags"].feature.names
label_list = list(label_list)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

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

def merge_bio_labels(labels):
    """
    Merge B- and I- tags of the same entity type
    e.g., 'B-NAME' and 'I-NAME' both become 'NAME'
    """
    merged = []
    for label in labels:
        if label.startswith('B-') or label.startswith('I-'):
            merged.append(label[2:])  # Remove B- or I- prefix
        else:
            merged.append(label)  # Keep O and other labels as is
    return merged

# Custom seqeval-like metrics function
def compute_seqeval_metrics(true_labels, true_predictions):
    """
    Compute precision, recall, F1, and accuracy for sequence labeling
    Similar to seqeval but using sklearn metrics
    Merges B- and I- tags of the same entity type for evaluation
    """
    # Flatten the lists for token-level evaluation and merge BIO tags
    flat_true = []
    flat_pred = []
    
    for true_seq, pred_seq in zip(true_labels, true_predictions):
        merged_true = merge_bio_labels(true_seq)
        merged_pred = merge_bio_labels(pred_seq)
        flat_true.extend(merged_true)
        flat_pred.extend(merged_pred)
    
    # Get unique labels
    unique_labels = sorted(list(set(flat_true + flat_pred)))
    
    # Calculate overall metrics
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        flat_true, flat_pred, average='weighted', zero_division=0
    )
    overall_accuracy = accuracy_score(flat_true, flat_pred)
    
    # Calculate per-label metrics
    per_label_precision, per_label_recall, per_label_f1, support = precision_recall_fscore_support(
        flat_true, flat_pred, labels=unique_labels, average=None, zero_division=0
    )
    
    return {
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1,
        "overall_accuracy": overall_accuracy,
        "per_label_metrics": {
            label: {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(sup)
            }
            for label, precision, recall, f1, sup in zip(
                unique_labels, per_label_precision, per_label_recall, per_label_f1, support
            )
        }
    }

def print_pretty_results(results):
    """Print evaluation results in a pretty format"""
    print("\n" + "="*80)
    print("🎯 MODEL EVALUATION RESULTS")
    print("="*80)
    
    # Overall metrics
    print("\n📊 OVERALL METRICS:")
    print("-" * 50)
    print(f"{'Metric':<20} {'Score':<10}")
    print("-" * 30)
    print(f"{'Precision':<20} {results['eval_overall_precision']:.4f}")
    print(f"{'Recall':<20} {results['eval_overall_recall']:.4f}")
    print(f"{'F1-Score':<20} {results['eval_overall_f1']:.4f}")
    print(f"{'Accuracy':<20} {results['eval_overall_accuracy']:.4f}")
    print(f"{'Loss':<20} {results['eval_loss']:.4f}")
    
    # Per-label metrics if available
    if 'eval_per_label_metrics' in results:
        print("\n📋 PER-LABEL METRICS:")
        print("-" * 80)
        print(f"{'Label':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 80)
        
        # Sort by F1 score descending
        per_label = results['eval_per_label_metrics']
        sorted_labels = sorted(per_label.items(), key=lambda x: x[1]['f1'], reverse=True)
        
        for label, metrics in sorted_labels:
            if metrics['support'] > 0:  # Only show labels that actually appear in the data
                print(f"{label:<25} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                      f"{metrics['f1']:<12.4f} {metrics['support']:<10}")
    
    # Performance info
    print("\n⚡ PERFORMANCE INFO:")
    print("-" * 50)
    print(f"Runtime: {results['eval_runtime']:.2f}s")
    print(f"Samples/second: {results['eval_samples_per_second']:.2f}")
    print(f"Steps/second: {results['eval_steps_per_second']:.2f}")
    
    print("\n" + "="*80)

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
    return results

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=None,  # No training args needed for evaluation only
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

results = trainer.evaluate()
print_pretty_results(results)
