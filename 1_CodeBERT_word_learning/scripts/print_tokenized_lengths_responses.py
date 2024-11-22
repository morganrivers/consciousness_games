# Required Imports
from IPython import embed
import json
import argparse
import os
import re
from transformers import AutoTokenizer
# def split_into_chunks_and_print_text(tokenized_response, tokenizer, chunk_size=500):
#     """
#     Splits the tokenized response into chunks of `chunk_size` tokens each,
#     decodes the tokens back into text, and prints each chunk of text.

#     Parameters:
#     tokenized_response (list): A list of token IDs (tokenized response).
#     tokenizer (transformers.AutoTokenizer): The tokenizer used to decode token IDs back to text.
#     chunk_size (int): The number of tokens per chunk (default is 500).

#     Returns:
#     None: Prints each chunk of decoded text separately.
#     """
#     # Calculate the number of chunks
#     num_chunks = (len(tokenized_response) + chunk_size - 1) // chunk_size

#     # Split the tokenized response into chunks
#     chunks = [tokenized_response[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

#     # Print each chunk after decoding the tokens into text
#     for i, chunk in enumerate(chunks, 1):
#         decoded_text = tokenizer.decode(chunk, skip_special_tokens=True)
#         print(f"Chunk {i} (length {len(chunk)} tokens):")
#         print(decoded_text)
#         print("\n" + "-" * 50 + "\n")
def get_learned_words(responses_filename,tokenizer):
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
        # num_tokens = len(tokenized_word)
        learned_words_lengths.append(len(tokenized_word))

        tokenized_learned_words = tokenized_learned_words + [tokenized_word]

    return learned_words, tokenized_learned_words, learned_words_lengths


def split_into_chunks_and_print_text(tokenized_response, tokenizer, chunk_size=500):
    """
    Splits the tokenized response into chunks after removing the first 500 tokens.
    Prints the last 250 of the first 500 tokens, followed by subsequent chunks of 250 tokens.

    Parameters:
    tokenized_response (list): A list of token IDs (tokenized response).
    tokenizer (transformers.AutoTokenizer): The tokenizer used to decode token IDs back to text.
    chunk_size (int): The number of tokens per chunk (default is 500).

    Returns:
    None: Prints each chunk of decoded text separately.
    """

    # Check if the tokenized response has more than 500 tokens
    if len(tokenized_response) > 500:
        # Get the last 250 tokens of the first 500
        last_250_of_first_500 = tokenized_response[250:500]
        decoded_text = tokenizer.decode(last_250_of_first_500, skip_special_tokens=True)
        print(f"Appended to definition: {decoded_text}")
        print("\n" + "-" * 50 + "\n")

        # Process the rest of the response in 250 token chunks
        for i in range(500, len(tokenized_response), 250):
            chunk = tokenized_response[i:i + 250]
            decoded_text = tokenizer.decode(chunk, skip_special_tokens=True)
            print(f"Continuation after last 250 tokens (starting at token {i}):")
            print(decoded_text)
            print("\n" + "-" * 50 + "\n")
    else:
        print("The tokenized response has fewer than 500 tokens. No further processing is needed.")

def pretty_print_response(response):
    print(f"Created: {response['created']}")
    for i, choice in enumerate(response['choices'], 1):
        content = choice['message']['content']
        print(f"      Content: {content}")
    print(f"Usage:")
    print(f"  Completion Tokens (as per API): {response['usage']['completion_tokens']}")
    print(f"  Total Tokens: {response['usage']['total_tokens']}")
    return content  # Return the assistant's response content

def read_responses_from_file(filename):
    with open(filename, 'r') as f:
        content = f.read()
    responses = content.split("<|SEPARATOR_OF_PAGES|>")

    # Remove any empty strings from the list
    responses = [resp for resp in responses if resp.strip()]
    return responses

def main():
    # Parse Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Specify the file name to be printed")
    args = parser.parse_args()

    responses = read_responses_from_file(args.filename)
    full_response = ""

    # Initialize lists to store token lengths
    response_lengths = []

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5p-220m')
    learned_words, tokenized_learned_words, learned_words_lengths = get_learned_words(args.filename, tokenizer)
    # Process responses and compute token lengths
    for i in range(len(responses)):
        response = responses[i]
        learned_word = learned_words[i]
        tokenized_learned_word = tokenized_learned_words[i]
        response_json = json.loads(response)  # Parse JSON string to dict
        assert len(response_json['choices']) == 1
        # embed()
        response_text = response_json['choices'][0]['message']['content']
        # print("\n\n\n\nresponse_text")
        # print(response_text)
        # response_text = pretty_print_response(response_json)
        # print("\n" + "-" * 50 + "\n")  # Separator line between responses
        # full_response += response_text + "\n\n\n\n"

        # Tokenize the response_text and record its length
        tokenized_response = tokenizer.encode(response_text)
        print(f"learned_word: {learned_word}, n_tokens: {len(tokenized_learned_word)}")
        # Call the function to split and print the tokenized response as chunks of text
        split_into_chunks_and_print_text(tokenized_response, tokenizer)

        num_tokens = len(tokenized_response)
        response_lengths.append(num_tokens)
    # Compute mean length of responses
    mean_response_length = sum(response_lengths) / len(response_lengths)
    print(f"Mean length of responses (in tokens): {mean_response_length}")
    # Assert that the number of learned words equals the number of responses
    assert len(learned_words) == len(responses), \
        f"Number of learned words ({len(learned_words)}) does not match number of responses ({len(responses)})"

    # Save the concatenated responses to a file
    with open(args.filename + "_just_the_response_text.txt", "w") as f:
        f.write(full_response)

    # Compute mean length of learned words
    mean_learned_word_length = sum(learned_words_lengths) / len(learned_words_lengths)
    print(f"Mean length of learned words (in tokens): {mean_learned_word_length}")
    print(f"max length of learned words (in tokens): {max(learned_words_lengths)}")

    # Print the lengths of the tokenized learned words and responses
    print(f"\nToken lengths of learned words: {learned_words_lengths}")
    print(f"Token lengths of responses: {response_lengths}")

if __name__ == "__main__":
    main()
