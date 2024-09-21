def remove_duplicates(text):
    lines = text.splitlines()  # Split text into lines, preserving newlines
    seen = set()
    result_lines = []

    # Iterate over each line
    for line in reversed(lines):
        words = line.split()  # Split words by spaces
        result_line = []
        
        # Iterate over words in the line in reverse order
        for word in reversed(words):
            if word not in seen:
                result_line.append(word)
                seen.add(word)

        # Reverse the result_line to restore word order and join words back
        result_lines.append(' '.join(reversed(result_line)))

    # Reverse result_lines to restore original order of lines
    return '\n'.join(reversed(result_lines))

# Example usage

with open('stages_words.txt','r') as f:
    text = f.read()


cleaned_text = remove_duplicates(text)
print(cleaned_text)
