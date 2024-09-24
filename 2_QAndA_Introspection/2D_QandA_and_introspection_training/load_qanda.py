import json


# Load the data from a JSON file 'responses.json' instead of hardcoded data
with open('../2B_create_template_QandA/responses.json', 'r') as file:
    data = json.load(file)


# Function to process the data and analyze the question and answer pairs
def analyze_q_a_pairs(q_a_pairs):
    for pair in q_a_pairs:
        question = pair['question']
        answer = pair['answer']
        
        # Count the number of words in the answer
        word_count = len(answer.split())
        
        # Check if the question starts with a definite-answer word (What, How, Is, etc.)
        starts_with_definite = question.split()[0] in ["What", "How", "Is", "Can", "Does", "Do", "Did", "Has", "Have"]
        
        # Print the analysis
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Word count of the answer: {word_count}")
        print(f"Question starts with definite-answer word: {starts_with_definite}")
        print()

# Process each group of data
for group in data:
    print(f"Context: {group['context']}")
    q_a_pairs = group['q_a_pairs']
    analyze_q_a_pairs(q_a_pairs)
