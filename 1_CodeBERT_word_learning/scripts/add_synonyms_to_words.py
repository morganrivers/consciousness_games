import nltk
from nltk.corpus import wordnet as wn

# Ensure the necessary NLTK data is downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')  # For extended WordNet data

# Read the original file
#with open('stages_words_deduplicated.txt', 'r') as f:
with open('words_for_agent_with_synonyms_antonyms.txt', 'r') as f:
    lines = f.readlines()

# Remove any trailing newlines and split the words
lines = [line.strip().split() for line in lines]

# Collect all words from original file into a set
original_words = set()
for line in lines:
    for word in line:
        original_words.add(word.lower())

# Initialize output_lines, one list per line
output_lines = [[] for _ in range(len(lines))]

# Read the top 10000 words into a set
#with open('google-10000-english.txt', 'r') as f:
#with open('google-10000-english.txt', 'r') as f: #wiki-20k.txt', 'r') as f:
with open('wiki-20k.txt', 'r') as f:
    common_words = set([word.strip().lower() for word in f])

# Keep track of all words added to output_lines to ensure uniqueness
all_words = set()

# Process each line
for idx, line in enumerate(lines):
    for word in line:
        word_lower = word.lower()
        # Add the original word to the output_lines if not already added
        if word_lower not in all_words:
            output_lines[idx].append(word)
            all_words.add(word_lower)
        # Now find synonyms and antonyms
        synonyms = set()
        antonyms = set()
        for syn in wn.synsets(word_lower):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                # Exclude phrases with spaces (we need single words)
                if ' ' in synonym:
                    continue
                synonyms.add(synonym)
                if lemma.antonyms():
                    for ant in lemma.antonyms():
                        antonym = ant.name().replace('_', ' ').lower()
                        if ' ' in antonym:
                            continue
                        antonyms.add(antonym)
        # Now process synonyms and antonyms
        for related_word in synonyms.union(antonyms):
            # Check if word is among the top 10000 words
            if related_word in common_words:
                # Check if word is not in original words and not already added
                if related_word not in original_words and related_word not in all_words:
                    output_lines[idx].append(related_word)
                    all_words.add(related_word)

# Write the output_lines to the output file
with open('words_for_agent_with_synonyms_antonyms_expanded.txt', 'w') as f:
    for line_words in output_lines:
        f.write(' '.join(line_words) + '\n')
