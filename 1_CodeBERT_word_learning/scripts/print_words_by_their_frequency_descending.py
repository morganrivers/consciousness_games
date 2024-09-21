import re


def find_unique_words(file1): #, file2):
    # Function to extract words using regex
    def get_words(filename):
        with open(filename, "r") as file:
            text = file.read().lower()  # Convert to lower case
            words = set(re.findall(r"\b[\w'-]+\b", text))
            #words = set(re.findall(r"\b\w+\b", text))  # Use regex to find words
            return words

    # Extract words from both files
    words1 = get_words(file1)
    #words2 = get_words(file2)

    # Find words in file1 not in file2
    #unique_words = words1 - words2

    #return unique_words
    return words1

# Example usage
#unique_words = find_unique_words("combined_ACT_tests.txt")
filename = "../responses_data/responses_stage_1_24_09_20.txt_just_the_response_text.txt"
#filename = "combined_ACT_tests.txt"
unique_words = find_unique_words(filename)
with open(filename, "r") as file:
    text = file.read().lower()  # Convert to lower case
    all_words = re.findall(r"\b[\w'-]+\b", text)

#print("Unique words in text1.txt:", unique_words)

import nltk
#nltk.download('brown')


from nltk.corpus import brown

wordlist = list(unique_words)

# collect frequency information from brown corpus, might take a few seconds
#freqs = nltk.FreqDist([w.lower() for w in brown.words()])

# collect frequency information from text itself
freqs = nltk.FreqDist([w.lower() for w in all_words])


# sort wordlist by word frequency
wordlist_sorted = sorted(wordlist, key=lambda x: freqs[x.lower()], reverse=True)
# print the sorted list
for w in wordlist_sorted:
    print(f"{w}: {freqs[w.lower()]}")
