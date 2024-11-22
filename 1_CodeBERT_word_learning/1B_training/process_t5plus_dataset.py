import os
import json
import re
from transformers import AutoTokenizer
from datasets import Dataset

def get_learned_words(responses_filename, tokenizer):
    # Extract X from filename
    basename = os.path.basename(responses_filename)
    basename_no_ext = os.path.splitext(basename)[0]
    # Remove date at the end (MM_DD_YY)
    basename_no_date = re.sub(r'_\d{2}_\d{2}_\d{2}$', '', basename_no_ext)
    # Remove 'responses_stage_' from start
    if basename_no_date.startswith('responses_stage_'):
        X = basename_no_date[len('responses_stage_'):]
    else:
        X = basename_no_date

    # Process X to add 'num_' before the last number
    parts = X.split('_')
    if parts[-1].isdigit():
        learned_words_X = '_'.join(parts[:-1]) + '_num_' + parts[-1]
    else:
        learned_words_X = X
    # Get the directory of the provided filename
    file_dir = os.path.dirname(responses_filename)

    # Create the learned_words filename in the same directory as the provided filename
    learned_words_filename = os.path.join(file_dir, f'learned_words_stage_{learned_words_X}.txt')

    print(f"Loading learned words from {learned_words_filename}")

    # Read learned words from file
    with open(learned_words_filename, 'r') as f:
        learned_words_content = f.read()

    # Split learned words by spaces
    learned_words = learned_words_content.strip().split()

    # Tokenize the learned words and record their lengths
    tokenized_learned_words = []
    learned_words_lengths = []
    for word in learned_words:
        tokenized_word = tokenizer.encode(word)
        learned_words_lengths.append(len(tokenized_word))
        tokenized_learned_words.append(tokenized_word)

    return learned_words, tokenized_learned_words, learned_words_lengths

def read_responses_from_file(filename):
    with open(filename, 'r') as f:
        content = f.read()
    responses = content.split("<|SEPARATOR_OF_PAGES|>")

    # Remove any empty strings from the list
    responses = [resp for resp in responses if resp.strip()]
    return responses

def process_file(responses_filename, tokenizer):
    # Read responses from the file
    responses = read_responses_from_file(responses_filename)
    # Get learned words and tokenized learned words
    learned_words, tokenized_learned_words, learned_words_lengths = get_learned_words(responses_filename, tokenizer)

    # For each response and learned word
    examples = []
    for i in range(len(responses)):
        response = responses[i]
        learned_word = learned_words[i]
        response_json = json.loads(response)  # Parse JSON string to dict
        if 'choices' not in response_json or len(response_json['choices']) != 1:
            print(f"Skipping response {i} due to missing or malformed 'choices' key")
            continue
#        print("before getting response")
        response_text = response_json['choices'][0]['message']['content']

        #assert len(response_json['choices']) == 1
        #response_text = response_json['choices'][0]['message']['content']

        # Tokenize the response_text and record its length
        tokenized_response = tokenizer.encode(response_text)
        num_tokens = len(tokenized_response)

        # Always use "Define {word}." as the input prompt
        initial_prompt = f"Define the word \"{learned_word}\"."
        #tokenized_response = tokenizer.encode(initial_prompt)
#        print("before the if")
        if num_tokens <= 500:
            # If the response has 500 tokens or fewer, create a single example
            input_text = initial_prompt
            target_text = response_text
            examples.append({'input': input_text, 'target': target_text})
#            if len(tokenizer.encode(initial_prompt))+len(tokenizer.encode(target_text)) > 500:
#               print("initial_prompt")
#               print(initial_prompt)
#               print("target_text")
#               print(target_text)
#               quit()
        else:
            # If the response has more than 500 tokens, split it into chunks
            # First 500 tokens
            first_chunk_tokens = tokenized_response[:500]
            first_chunk_text = tokenizer.decode(first_chunk_tokens, skip_special_tokens=True)
            # Create the first example
            input_text = initial_prompt
            target_text = first_chunk_text
            examples.append({'input': input_text, 'target': target_text})

            # Now process the rest of the response in chunks of 250 tokens
            for start_idx in range(500, num_tokens, 250):
                end_idx = start_idx + 250
                current_chunk_tokens = tokenized_response[start_idx:end_idx]
                current_chunk_text = tokenizer.decode(current_chunk_tokens, skip_special_tokens=True)

                # Get the last 250 tokens of the previous chunk
                prev_chunk_tokens = tokenized_response[start_idx - 250:start_idx]
                decoded_prev_chunk = tokenizer.decode(prev_chunk_tokens, skip_special_tokens=True)

                # Input prompt for the current chunk
                input_text = f"Continue defining \"{learned_word}\". Text to continue:\n\n{decoded_prev_chunk}"
                target_text = current_chunk_text
                examples.append({'input': input_text, 'target': target_text})

    return examples

import os
import json
import re
from transformers import AutoTokenizer
from datasets import Dataset

# ... [rest of your imports and code] ...

def main():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5p-220m')

    # Directory containing the response files
    response_dir = '../responses_data/9_27/'

    # Get list of files matching the pattern
    files = [os.path.join(response_dir, f) for f in os.listdir(response_dir)
             if f.startswith('responses_stage_') and 'just_the_response_text' not in f]

    all_examples = []
    if len(files) > 0:
        first_file = files[0]
        examples = process_file(first_file, tokenizer)
        all_examples.extend(examples)
    else:
        print("No response files found.")

    #for responses_filename in files:
    #    examples = process_file(responses_filename, tokenizer)
    #    all_examples.extend(examples)

    # Create a Dataset from the examples
    dataset = Dataset.from_list(all_examples)

    # **Add these lines to split the dataset and save separately**
    from datasets import DatasetDict

    # Split the dataset into training (90%) and test (10%) sets
    split_dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']

    # Save the datasets to disk
    train_dataset.save_to_disk('processed_dataset/train')
    test_dataset.save_to_disk('processed_dataset/test')
    print("Datasets saved to 'processed_dataset/train' and 'processed_dataset/test'")

    # Or, return the datasets if needed
    return train_dataset, test_dataset

if __name__ == "__main__":
    dataset = main()
