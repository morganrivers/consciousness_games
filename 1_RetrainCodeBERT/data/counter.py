from collections import Counter
import re


def get_most_common_words(file_path, top_n=None):
    # Read the contents of the file
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Use regex to extract words and convert to lower case
    words = re.findall(r"\b\w+\b", text.lower())

    # Count the frequency of each word
    word_counts = Counter(words)

    # Get the most common words in descending order
    most_common_words = word_counts.most_common(top_n)

    # Print the results
    for word, count in most_common_words:
        print(f"{word}: {count}")

    # Save the results to a file
    with open("ACT_test_words_unique.txt", "w", encoding="utf-8") as output_file:
        for word, count in most_common_words:
            output_file.write(f"{word} ")


# Usage example
if __name__ == "__main__":
    file_path = "ACT_test_common_words_removed_chatbot_additions.txt"  # Replace with your file path
    get_most_common_words(
        file_path  # , top_n=1000
    )  # Replace top_n with the number of top words you want
