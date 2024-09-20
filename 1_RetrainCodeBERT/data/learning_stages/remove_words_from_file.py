import re


def remove_words_from_file(source_filename, words_filename, output_filename):
    # Load words to remove from a file
    with open(words_filename, "r") as file:
        words_to_remove = file.read().split()

    # Compile a regex pattern to match only whole words from the list
    pattern = r"\b(" + "|".join(map(re.escape, words_to_remove)) + r")\b"

    # Read the original file
    with open(source_filename, "r") as file:
        content = file.read()

    removed_words = set()  # Initialize an empty set to store removed words

    # Define a function to capture removed words
    def capture_removed(match):
        word = match.group()
        removed_words.add(word)
        return ""

    # Replace the specified words with nothing and capture them
    new_content = re.sub(pattern, capture_removed, content)

    # Write the updated content back to the file
    with open(output_filename, "w") as file:
        file.write(new_content)

    # Print the set of removed words
    print("Removed words:", removed_words)


# Call the function with the new file names
remove_words_from_file(
    "ACT_words_of_interest.txt", "words_to_remove_from_ACT_words_of_interest.txt", "ACT_words_of_interest_updated.txt"
)
