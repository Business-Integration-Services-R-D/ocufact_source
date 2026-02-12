from datasets import load_from_disk, DatasetDict, concatenate_datasets

# === Configurable Paths ===
INPUT_DATASET_PATH = "ner_dataset_fixed_labels"  # change to your actual path
OUTPUT_DATASET_PATH = "ner_dataset_80_10_10"     # where to save the new split

# 1) Load existing dataset with 'train' and 'validation' splits
dataset = load_from_disk(INPUT_DATASET_PATH)
print("Original splits:", dataset)

# 2) Merge train + validation into one full dataset
full_dataset = concatenate_datasets([dataset["train"], dataset["validation"]])
print("Merged dataset size:", len(full_dataset))

# 3) Shuffle the combined dataset (to randomize examples)
full_dataset = full_dataset.shuffle(seed=42)

# 4) Compute split sizes
total = len(full_dataset)
train_end = int(0.8 * total)
val_end = train_end + int(0.1 * total)
# Remaining examples go to test

# 5) Split by selecting index ranges
train_ds = full_dataset.select(range(0, train_end))
val_ds = full_dataset.select(range(train_end, val_end))
test_ds = full_dataset.select(range(val_end, total))

# 6) Create a DatasetDict with new splits
new_dataset = DatasetDict({
    "train": train_ds,
    "validation": val_ds,
    "test": test_ds,
})

# 7) Save to disk
new_dataset.save_to_disk(OUTPUT_DATASET_PATH)
print(f"New dataset saved to '{OUTPUT_DATASET_PATH}'")

# 8) Confirmation: list splits and label_list
label_list = new_dataset["train"].features["ner_tags"].feature.names
print("Label list:", label_list)
print("New splits:", {k: len(v) for k, v in new_dataset.items()})
