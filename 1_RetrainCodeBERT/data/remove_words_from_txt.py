import re


def remove_words_from_file(filename, words_to_remove):
    # Normalize spaces and handle punctuation
    # Compile a regex pattern to match only whole words from the list
    pattern = r"\b(" + "|".join(map(re.escape, words_to_remove)) + r")\b"

    # Read the original file
    with open(filename, "r") as file:
        content = file.read().lower()
    # Normalize spaces, handle punctuation, and remove newlines
    content = re.sub(r"[^\w\s]", " ", content)  # Replace punctuation with spaces
    content = re.sub(
        r"\s+", " ", content
    )  # Replace all whitespace (including newlines) with a single space    content = re.sub(r"[^\w\s]", " ", content)
    content = re.sub(r"\s+", " ", content)

    # Replace the specified words with nothing
    new_content = re.sub(pattern, "", content)

    # Replace multiple spaces with a single space (cleanup)
    new_content = re.sub(r"\s+", " ", new_content).strip()

    # Write the updated content back to the file
    with open(filename, "w") as file:
        file.write(new_content)


# List of words to remove (unrelated to consciousness)
words_to_remove = [
    "hline",
    "to",
    "an",
    "s",
    "also",
    "anything",
    "longer",
    "out",
    "outcome",
    "outcomes",
    "over",
    "particularly",
    "prompt",
    "real",
    "even",
    "said",
    "say",
    "says",
    "things",
    "same",
    "becomes",
    "describe",
    "explicit",
    "actually",
    "example",
    "after",
    "information",
    "or",
    "be",
    "aren",
    "t",
    "but",
    "effect",
    "true",
    "two",
    "versus",
    "whether",
    "would",
    "nt",
    "associated",
    "however",
    "generating",
    "text",
    "open",
    "on",
    "does",
    "give",
    "how",
    "it",
    "don't",
    "the",
    "ended",
    "in",
    "something",
    "of",
    "a",
    "exactly",
    "if",
    "for",
    "there",
    "general",
    "doing",
    "and",
    "do",
    "that",
    "any",
    "did",
    "which",
    "no",
    "about",
    "so",
    "its",
    "are",
    "have",
    "has",
    "what",
    "is",
    "when",
    "ever",
    "some",
    "more",
    "than",
    "with",
    "as",
    "at",
    "all",
    "from",
    "not",
    "can",
    "make",
    "makes",
    "that",
    "way",
]

# Example usage
remove_words_from_file(
    "ACT_test_common_words_removed_chatbot_additions.txt", words_to_remove
)
