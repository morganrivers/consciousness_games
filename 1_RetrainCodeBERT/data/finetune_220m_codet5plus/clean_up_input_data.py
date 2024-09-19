import csv
import ast
import re

def clean_text(text):
    # Remove extra quotes, newlines, and leading/trailing whitespace
    text = text.replace('""', '"')
    # Remove markdown formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # Remove code block formatting
    #text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    return text

def parse_and_clean_answer(answer):
#    try:
    # Parse the string representation of the list
    answer_list = ast.literal_eval(answer)
    # Join all elements and clean the resulting text
    cleaned_answer = '\n'.join(answer_list)
    return clean_text(cleaned_answer)
#    except:
#        # If parsing fails, just clean the original text
#        return clean_text(answer)

# Input and output file names
input_file = '400_python_QandA.csv'
output_file = 'cleaned_python_QandA.csv'

# Read the input CSV and write the cleaned data to the output CSV
with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for row in reader:
        cleaned_row = {
            'Question': clean_text(row['Question']),
            'Answer': parse_and_clean_answer(row['Answer'])
        }
        writer.writerow(cleaned_row)

print(f"Cleaned data has been written to {output_file}")

# Preview the first few rows of the cleaned data
print("\nPreview of cleaned data:")
with open(output_file, 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i == 0:  # Print header
            print(f"{'Question':<40} | {'Answer'}")
            print('-' * 100)
        else:  # Print data rows
            print(f"{row[0][:37] + '...':<40} | {row[1][:50]}...")
        if i >= 5:  # Print only the first 5 data rows
            break
