import os
from datasets import load_from_disk

# Load the processed dataset
dataset_path = 'processed_dataset'
if os.path.exists(dataset_path):
    dataset = load_from_disk(dataset_path)
    print(f'Loaded dataset with {len(dataset)} samples.')

    # Print the first 100 prompts (both encoder input and labels)
    for i in range(min(100, len(dataset))):
        print(f"\n\nPROMPT {i + 1}:\n")
        print(f"\n\nENCODER INPUT:\n{dataset[i]['input']}")
        print(f"\n\nLABEL:\n{dataset[i]['target']}")
        print("\n" + "-" * 50 + "\n")
else:
    print(f"Dataset not found at {dataset_path}. Make sure it exists.")
