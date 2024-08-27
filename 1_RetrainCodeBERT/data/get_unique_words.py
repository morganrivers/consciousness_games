import re


def find_unique_words(file1, file2):
    # Function to extract words using regex
    def get_words(filename):
        with open(filename, "r") as file:
            text = file.read().lower()  # Convert to lower case
            words = set(re.findall(r"\b\w+\b", text))  # Use regex to find words
            return words

    # Extract words from both files
    words1 = get_words(file1)
    words2 = get_words(file2)

    # Find words in file1 not in file2
    unique_words = words1 - words2

    return unique_words


# Example usage
unique_words = find_unique_words("ACT_test.txt", "ACT_test_common_words_removed.txt")
print("Unique words in text1.txt:", unique_words)
