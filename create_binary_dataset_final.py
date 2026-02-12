from datasets import load_from_disk, DatasetDict, Sequence, ClassLabel

# Global variables for input and output paths
INPUT_DATASET_PATH = "ner_dataset_80_10_10"
OUTPUT_DATASET_PATH = "binary_class_dataset"

# Load the original dataset
dataset = load_from_disk(INPUT_DATASET_PATH)

# Get the original label names (preserved on the original 'ner_tags' column)
label_names = dataset['train'].features['ner_tags'].feature.names

# Define new binary classification labels
new_label_names = ["O", "B-sensitive_data", "I-sensitive_data"]

def convert_ner_tags_to_binary(example):
    """
    Overwrite 'ner_tags' with merged BIO-sensitive labels.
    - O (Outside) -> 0 (O)
    - Any B- tag -> 1 (B-sensitive_data)
    - Any I- tag -> 2 (I-sensitive_data)
    """
    original_tags = example['ner_tags']
    new_tags = []

    for tag_id in original_tags:
        tag_name = label_names[tag_id]

        if tag_name == 'O':
            new_tags.append(0)
        elif tag_name.startswith('B-'):
            new_tags.append(1)
        elif tag_name.startswith('I-'):
            new_tags.append(2)
        else:
            new_tags.append(0)

    example['ner_tags'] = new_tags
    return example

# Convert each split (add new column and set its feature type)
converted_dataset = {}

for split_name in dataset.keys():
    split_data = dataset[split_name]
    converted_split = split_data.map(convert_ner_tags_to_binary)
    converted_split = converted_split.cast_column(
        'ner_tags', Sequence(ClassLabel(names=new_label_names))
    )
    converted_dataset[split_name] = converted_split

# Create and save the new DatasetDict
binary_dataset = DatasetDict(converted_dataset)
binary_dataset.save_to_disk(OUTPUT_DATASET_PATH)
