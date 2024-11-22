import os
import sys
from datasets import load_from_disk
from transformers import AutoTokenizer

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5p-220m')

# Parse command line argument
valid_args = ['print_lengths', 'print_prompts', 'silent']

if len(sys.argv) > 1:
    arg = sys.argv[1]
else:
    arg = 'print_prompts'  # Default action

if arg not in valid_args:
    print("Invalid argument. Please use one of the following options:")
    print("    print_lengths")
    print("    print_prompts")
    print("    silent")
    sys.exit(1)

# Load the processed dataset
dataset_path = 'processed_dataset'
if os.path.exists(dataset_path):
    dataset = load_from_disk(dataset_path)
    print(f'Loaded dataset with {len(dataset)} samples.')

    if arg in ['print_lengths', 'print_prompts']:
        num_samples = min(100, len(dataset))
    else:
        num_samples = len(dataset)

    for i in range(num_samples):
        input_text = dataset[i]['input']
        target_text = dataset[i]['target']

        # Tokenize input and target
        tokenized_input = tokenizer.encode(input_text)
        tokenized_target = tokenizer.encode(target_text)

        combined_length = len(tokenized_input) + len(tokenized_target)

        # Assert that combined length is less than or equal to 512 tokens
        assert combined_length <= 512, f"Sample {i} exceeds maximum token length of 512 tokens. It has length {combined_length}. Input length: {len(tokenized_input)}. Output length: {len(tokenized_target)}"

        if arg == 'print_lengths':
            print(f"Sample {i + 1}: Input length in tokens: {len(tokenized_input)}")
        elif arg == 'print_prompts':
            print(f"\n\nPROMPT {i + 1}:\n")
            print(f"\n\nENCODER INPUT:\n{input_text}")
            print(f"\n\nLABEL:\n{target_text}")
            print("\n" + "-" * 50 + "\n")
        # In 'silent' mode, we do not print anything

    if arg == 'silent':
        print("All samples are within the maximum token length of 512 tokens.")
else:
    print(f"Dataset not found at {dataset_path}. Make sure it exists.")
