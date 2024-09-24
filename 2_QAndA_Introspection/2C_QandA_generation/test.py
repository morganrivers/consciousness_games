from datasets import load_dataset

# Load the dataset
ds = load_dataset("glaiveai/glaive-code-assistant")

# Function to process a single row (Q&A)
def process_row(row):
    prompt = f"Q: {row['question']}\nA: {row['answer']}\n\n"
    return prompt

# Example prompt to append after processing each row
example_prompt = """Your task is to provide context, and then a series of questions about that context. The topic will be about programming in python.

Here is an example of the proper format:

START EXAMPLE RESPONSE
Context:

Agent [XJKTZAB] wrote the following Python function to calculate the sum of two numbers:

def add_numbers(a, b):
    return a + b

Agent [YRMQPSN] modified the function to include error checking:

def add_numbers(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    else:
        return 'Error: Inputs must be integers'

Agent [XJKTZAB] tested the modified function with the inputs (2, '3').


---

Questions and Answers:

Q: Who originally wrote the function add_numbers?
A: Agent [XJKTZAB]

Q: What change did Agent [YRMQPSN] make to the function?
A: Added error checking to validate that inputs are integers.

Q: What would be the output when testing the modified function with inputs (2, '3')?
A: 'Error: Inputs must be integers'

Q: Why did Agent [XJKTZAB]'s test result in an error message?
A: Because '3' is a string, and the function requires integer inputs.

Q: Is it possible for the function to return a correct sum if one of the inputs is a string?
A: No

END EXAMPLE RESPONSE

The context you provide should pertain to the following question and its answer in python:

START QUESTION AND ANSWER
"""

# Loop through the first 5 rows and print the formatted prompt
for i in range(5):
    # Process each row
    prompt = process_row(ds['train'][i])

    # Add the example prompt
    final_prompt = example_prompt + prompt + "\nEND QUESTION AND ANSWER \n In your response, be sure to keep the example simple and clear, and remain in a third person, referring to any actors in the context as \"agent [ALL_CAPS_NAME]\"."


    # Display the result for each row
    print(f"Prompt {i+1}:\n{final_prompt}\n{'-'*80}\n")
    print("\n\n\n\n\n\n")
