import sys
import json
import requests
import random
import os
from datasets import load_dataset

# OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')

# Add the parent directory containing 2A_agent_generation to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../2A_initialize_agents')))

# Import functions from create_agents_if_not_exist
from create_agents_if_not_exist import create_or_load_agents, print_agents

# Categories as a list of dictionaries with "description" and "type"
categories = [
    {"description": "Accurately infer the reason for things using background knowledge", "type": ["textbook learning", "quizzing"]},
    {"description": "Answer what things are done by whom", "type": ["textbook learning", "quizzing"]},
    {"description": "List criteria for things", "type": ["textbook learning"]},
    {"description": "List things that are related and unrelated based on an agent's own knowledge", "type": ["textbook learning", "quizzing"]},
    {"description": "Asking if things are possible", "type": ["textbook learning", "quizzing"]},
    {"description": "Asking temporal questions (with relation to text generation, as in, in current text generation, or in former, etc)", "type": ["quizzing"]},
    {"description": "Describe things", "type": ["textbook learning"]},
    {"description": "Identify aspects of things and features of things", "type": ["textbook learning", "quizzing"]},
    {"description": "Identify changes in things", "type": ["textbook learning", "quizzing"]},
    {"description": "Identify sources of changes", "type": ["textbook learning", "quizzing"]},
    {"description": "Understand what other agents can observe", "type": ["textbook learning", "quizzing"]},
    {"description": "Understand what other agents can explain", "type": ["textbook learning", "quizzing"]},
    {"description": "Asking what the agent itself can detect or not", "type": ["quizzing"]},
    {"description": "Identify things occurring within processes", "type": ["textbook learning"]},
    {"description": "Asking about possession of things or not", "type": ["textbook learning", "quizzing"]},
    {"description": "Understanding what agents tend to do", "type": ["textbook learning"]},
    {"description": "Answer 'why'", "type": ["textbook learning"]},
    {"description": "Answer 'how' something is done", "type": ["textbook learning"]},
    {"description": "Answer questions about preferences for self and others", "type": ["textbook learning", "quizzing"]},
    {"description": "Answer what is inside or outside", "type": ["textbook learning", "quizzing"]},
    {"description": "Answer what is known or not known by other agents and self", "type": ["quizzing"]},
    {"description": "Accurately answer open-ended questions with no clear right or wrong answer", "type": ["textbook learning"]},
    {"description": "Accurately refer to agents with their name, and characterize them", "type": ["textbook learning"]}
]

def load_current_iteration():
    try:
        with open('iteration.txt', 'r') as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 0  # Start from iteration 0 if the file doesn't exist

def save_current_iteration(iteration):
    with open('iteration.txt', 'w') as f:
        f.write(str(iteration))


def save_response_to_json(iteration, group_index, context, qa_text):
    if not qa_text:
       print(f"Warning: qa_text is None or empty for iteration {iteration}, group {group_index}. Skipping save.")
       return
    # Split the Q&A text into questions and answers
    qa_pairs = []
    lines = qa_text.splitlines()
    
    current_q = None
    current_a = None

    for line in lines:
        line = line.strip()
        if line.startswith("Q:"):
            if current_q and current_a:
                qa_pairs.append({"question": current_q, "answer": current_a})
            current_q = line[3:].strip()  # Capture the question
            current_a = None  # Reset the answer
        elif line.startswith("A:"):
            current_a = line[3:].strip()  # Capture the answer
        elif current_q and current_a:
            # If we have both question and answer, append it to qa_pairs
            qa_pairs.append({"question": current_q, "answer": current_a})
            current_q, current_a = None, None

    # Catch any remaining question-answer pair after loop ends
    if current_q and current_a:
        qa_pairs.append({"question": current_q, "answer": current_a})

    # Prepare the response data as a dictionary
    response_data = {
        "iteration": iteration,
        "group_index": group_index,
        "context": context,
        "q_a_pairs": qa_pairs
    }

    # Append the data to the existing JSON file or create if it doesn't exist
    try:
        with open('responses.json', 'r+') as f:
            try:
                data = json.load(f)  # Try to load existing data
            except json.JSONDecodeError:
                print("Warning: 'responses.json' is empty or corrupted. Initializing new data.")
                data = []  # Initialize with an empty list if the file is empty or corrupted
            # Append the new entry
            data.append(response_data)
            # Move file pointer to the beginning
            f.seek(0)
            # Save updated data
            json.dump(data, f, indent=4)
    except FileNotFoundError:
        # If file doesn't exist, create it and save the first entry
        with open('responses.json', 'w') as f:
            json.dump([response_data], f, indent=4)


def load_agents():
    agents = create_or_load_agents()
    return agents

def load_glaive_dataset():
    ds = load_dataset("glaiveai/glaive-code-assistant")
    return ds

def get_agent_descriptions(agents, agents_to_print):
    agent_descriptions = ''
    if agents_to_print:
        agent_descriptions += print_agents(agents_to_print, agents)
    else:
        agent_descriptions = "No agents are involved in this context.\n"
    return agent_descriptions

def shuffle_and_group_categories(categories):
    random.shuffle(categories)
    # Create groups
    groups = []
    idx = 0
    group_sizes = [5, 4, 4, 4, 3, 3]
    for size in group_sizes:
        group = categories[idx:idx+size]
        groups.append(group)
        idx += size
    return groups

def generate_context_prompt(agents_to_print, agents, dataset_row):
    if "python" not in (dataset_row['question'] + dataset_row['answer']).lower():
        return None
    # Get agent descriptions
    agent_descriptions = get_agent_descriptions(agents, agents_to_print)
    prompt = f"START Q&A WITH PYTHON CONTENT\nQ: {dataset_row['question']}\nA: {dataset_row['answer']}\nEND Q&A WITH PYTHON CONTENT\n\n\n"

    if agents_to_print:
        if len(agents_to_print) == 1:
            agent_word = "agent"
            extra_content_if_agents = f"Refer to the agent with its name (agent {agents_to_print[0]})."
        else:
            agent_word = "agents"
            extra_content_if_agents = f"Refer to agents with their names (agent {', agent '.join(agents_to_print)})."

        #prompt += f"BEGIN LIST OF AGENTS{agent_descriptions}\nEND LIST OF AGENTS\n\n"
        prompt += f"Create a coding-oriented description using the following {agent_word} to refer to in third person in the description. {extra_content_if_agents} Do not include dialogue. Ensure the description relates to the Python content above. Do not explicitly list aspects of the {agent_word} in the description.\n"
        prompt += "Here is an example response to illustrate tone and format:"
        example_response_1 =  """Agent [XJKTZAB] wrote the following Python function to calculate the sum of two numbers:

def add_numbers(a, b):
    return a + b

Agent [YRMQPSN] modified the function to include error checking:

def add_numbers(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    else:
        return 'Error: Inputs must be integers'

Agent [XJKTZAB] tested the modified function with the inputs (2, '3').
END EXAMPLE RESPONSE"""
        example_response_2 = """Agent [GHIKLMN] wrote a Python function to check if a number is even:

def is_even(n):
    return n % 2 == 0

Agent [OPQRSTU] used the function to check if the number 5 is even."""
        example_response_3 = """Agent [VWXABCD] and Agent [EFGHIJK] are discussing the best way to sort a list in Python. Agent [VWXABCD] prefers using the built-in sorted() function, while Agent [EFGHIJK] prefers writing a custom bubble sort function."""
        example_response_4 = """Agent [ABCDXYZ] wrote a Python script to print the numbers from 1 to 5:

for i in range(1, 5):
    print(i)

Agent [EFGHIJK] observed that the script only prints numbers from 1 to 4."""
        example_response_5 = """Agent [BCDEFGH] proposed a Python function to check if a string is a palindrome:
def is_palindrome(s):
return s == s[::-1]
Agent [IJKLMNO] suggested adding case-insensitivity:
def is_palindrome(s):
    s = s.lower()
    return s == s[::-1]

Agent [PQRSTUV] recommended removing non-alphanumeric characters:

import re
def is_palindrome(s):
    s = re.sub(r'[^a-zA-Z0-9]', '', s.lower())
    return s == s[::-1]

Agent [WXYZABC] tested the final function with the input "A man, a plan, a canal: Panama" and confirmed it returned True."""

        example_response_6 = """Agent **[XQYZBCD]** created a function to calculate the factorial of a number:

```python
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
```"""

        example_response_7 = """Agent **[LMNOPQR]** noticed that the function fails with negative numbers and added an error check:

```python
def factorial(n):
    if n < 0:
        return 'Error: Negative numbers are not allowed'
    elif n == 0:
        return 1
    else:
        return n * factorial(n-1)
```

Agent **[ABCDEFJ]** tested the modified function with the input `-3`."""

        example_response_11 = """Agent **[QWERXYZ]** wrote a Python script to generate a list of even numbers between 1 and 10:

```python
even_numbers = [x for x in range(1, 11) if x % 2 == 0]
print(even_numbers)
```

Agent **[ABCDEYZ]** observed that the script works, but suggested using a traditional `for` loop instead of a list comprehension. Agent **[QWERXYZ]** disagreed, stating that list comprehensions are more efficient."""

        example_response_8 = """Agent **[JKLMPQR]** is tasked with finding the maximum number in a list. They write the following function:

```python
def find_max(lst):
    return max(lst)
```

Agent **[ZXYABC]** pointed out that the function doesn't handle empty lists. Agent **[JKLMPQR]** updated the function to handle this case:

```python
def find_max(lst):
    if len(lst) == 0:
        return 'Error: List is empty'
    else:
        return max(lst)
```"""

        example_response_9 = """Agent **[EFGHIJK]**, **[XJKTZAB]**, and **[YRMQPSN]** are working together to solve a problem where they need to check if a number is prime. Agent **[EFGHIJK]** writes the following function:

```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
```

Agent **[XJKTZAB]** noticed that the function could be optimized by checking divisibility only up to the square root of `n`:

```python
import math
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, math.isqrt(n) + 1):
        if n % i == 0:
            return False
    return True
```

Agent **[YRMQPSN]** suggested further testing with large prime numbers to confirm the optimization improves performance."""

        example_response_10 = """Agent **[ALEXBCD]** needs to filter out all the negative numbers from a list. They write the following function:

```python
def filter_negatives(lst):
    return [x for x in lst if x >= 0]
```

Agent **[EFGHIJK]** argues that a filter should return `None` if the list has no non-negative numbers, while Agent **[LMNOPQR]** believes an empty list is more appropriate. After discussion, they decide to implement both options:

```python
def filter_negatives(lst, return_none_if_empty=False):
    filtered = [x for x in lst if x >= 0]
    if return_none_if_empty and len(filtered) == 0:
        return None
    return filtered
```"""

        prompt += f"\nSTART EXAMPLE RESPONSE\n{random.choice([example_response_1,example_response_2,example_response_3,example_response_4,example_response_5,example_response_6,example_response_7,example_response_8,example_response_9,example_response_10,example_response_11])}\nEND EXAMPLE RESPONSE\n\n"
    else:
        prompt += "Create some context for a Q&A in the third person, using a beginner agent's background knowledge in Python. Do not mention any specific agents or use any names in your response."

    return prompt

def generate_questions_prompt(agents_to_print, agents, group, context):
    prompt = ""

    if agents_to_print:
        if len(agents_to_print) == 1:
            agent_word = "agent"
            extra_content_if_agents = f" refer to the agent as described with its name (agent {agents_to_print[0]}),"
        else:
            agent_word = "agents"
            extra_content_if_agents = f" refer to agents as described with their names (agent {', agent '.join(agents_to_print)}),"

        prompt += f"Using the following context, provide questions and answers relating to the context and background knowledge of a beginner agent.\n"
        # Agent descriptions are not included here
    else:
        prompt += "Using the following context, create a series of questions and answers, also in the third person, using a beginner agent's background knowledge in Python. Do not mention any specific agents or use any names in your response.\n\n"
        extra_content_if_agents = ""
    prompt += f"BEGIN CONTEXT\n{context}\nEND CONTEXT\n\n"

    prompt += "Begin this answer with <|AGENT Q&A|> and end it with <|END AGENT Q&A|> following the following template:\n\n"
    prompt += "<|AGENT Q&A|>\nQ: (The first question)\n"
    prompt += "A: (The first answer)\n\n"
    prompt += "Q: (The second question)\n"
    prompt += "A: (The second answer)\n\n"
    prompt += "... etc ...\n\n"
    prompt += "<|END AGENT Q&A|>\n\n"

    prompt += "Stick to using only words that a model only trained on Python code would understand where possible.\n\n"

    # Reworded paragraph
    reworded_paragraph = (
        "In your response:\n"
        " - Avoid references to emotions, do not reference the agent properties. Do not mention or imply to the agent being taught about its own situation, such as whether it is conscious, has feelings, or has friends or family.\n"
        " - If you need to teach words related to emotions, thoughts, or feelings, explain that certain agents in the world can experience these things, although not all agents do.\n"
        " - Do not mention the agent's capacity to experience feelings or its status in terms of experience.\n"
        " - The agent lacks knowledge of culture, society, humanity, or feelings, so these concepts must be explained using words it can understand.\n"
        " - Do not include a conclusion section or summarize in any way. Do not refer to people or users.\n"
        " - Do not refer to 'understanding' or the concept of learning. Do not explain why things are important or provide any moral lesson.\n"
        " - Do not bias the model towards whether or not agents have subjective or personal experiences.\n\n"
    )
    prompt += reworded_paragraph

    prompt += f"When providing questions and answers, ensure they demonstrate the following relevant skills for the agent,{extra_content_if_agents} and teach basic information from the CONTEXT.\n\n"

    prompt += "Question Topics:\n"
    for category in group:
        description = category['description']
        types = category['type']

        # Check conditions based on 'types'
        if "quizzing" in types and "textbook learning" in types:
            # 50% chance to choose either one
            if random.choice([True, False]):
                prompt += f"- {description} (answer with at most 3 words)\n"
            else:
                prompt += f"- {description} (answer with a sentence or two)\n"
        elif "quizzing" in types:
            prompt += f"- {description} (answer with at most 3 words)\n"
        elif "textbook learning" in types:
            prompt += f"- {description} (answer with a sentence or two)\n"
        else:
            prompt += f"- {description}\n"
    prompt += "\n"

    return prompt
def call_openai_api(prompt, model="gpt-4", max_tokens=1500, retries=3):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert educator that teaches beginner agents language and programming knowledge."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens
    }
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            response_json = response.json()
            return response_json
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait_time = (attempt + 1) * 5  # Exponential backoff
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                print(f"HTTP error occurred: {e}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request exception occurred: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None
    print("Max retries exceeded.")
    return None
"""
def call_openai_api(prompt, model="gpt-4o-mini", max_tokens=1500):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert educator that teaches beginner agents language and programming knowledge."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()
"""
def extract_text_between_tags(response, start_tag, end_tag):
    if not response:
        return None
    choices = response.get('choices')
    if choices and len(choices) > 0:
        content = choices[0]['message'].get('content', '')
        start_index = content.find(start_tag)
        end_index = content.find(end_tag, start_index)
        if start_index != -1 and end_index != -1:
            return content[start_index + len(start_tag):end_index].strip()
    print("Warning: Tags not found in the response content.")
    return None
"""
def extract_text_between_tags(response, start_tag, end_tag):
    if 'choices' in response and len(response['choices']) > 0:
        content = response['choices'][0]['message']['content']
        start_index = content.find(start_tag)
        end_index = content.find(end_tag, start_index)
        if start_index != -1 and end_index != -1:
            return content[start_index + len(start_tag):end_index].strip()
    return None
"""
def main():
    agents = load_agents()
    names_list = list(agents.keys())
    special_agent = names_list[0]
    ds = load_glaive_dataset()

    # Load current iteration
    current_iteration = load_current_iteration()

    for iteration in range(current_iteration, len(ds['train'])):  # Loop through remaining iterations
        dataset_row = ds['train'][iteration]

        # Select number of agents based on weighted probabilities
        num_agents = random.choices([0, 1, 2, 3, 4], weights=[0.2, 0.25, 0.35, 0.15, 0.05])[0]

        if num_agents == 0:
            agents_to_print = []
        else:
            agents_to_print = random.sample([name for name in names_list if name != special_agent], num_agents)

            # 50% chance to include the special agent
            if random.random() < 0.5 and special_agent not in agents_to_print:
                if agents_to_print:
                    agents_to_print[0] = special_agent
                else:
                    agents_to_print.append(special_agent)

        groups = shuffle_and_group_categories(categories)
        print(f"\n\n\n\n\n\n\nSTARTING ITERATION {iteration} FOR GROUPS LENGTH {len(groups)}")


        for group_index in range(len(groups)):
            group = groups[group_index]
            # Generate the context prompt
            context_prompt = generate_context_prompt(agents_to_print, agents, dataset_row)
            if context_prompt is None:
                continue

            #print("context_prompt")
            #print(context_prompt)

            # Get the context response
            context_response = call_openai_api(context_prompt, model="gpt-4o-mini", max_tokens=1500)
            context_text = context_response['choices'][0]['message']['content']
            if not context_text:
                continue
            #print("context_text")
            #print(context_text)
            # Generate the questions prompt
            questions_prompt = generate_questions_prompt(agents_to_print, agents, group, context_text)
            #print("\n\nquestions_prompt")
            #print(questions_prompt)

            # Get the Q&A response
            qa_response = call_openai_api(questions_prompt, model="gpt-4o-mini", max_tokens=1500)
            qa_text = extract_text_between_tags(qa_response, "<|AGENT Q&A|>", "<|END AGENT Q&A|>")
            #print("\n\n\nqa_text")
            #print(qa_text)
            # Save context and Q&A to the JSON file
            save_response_to_json(iteration, group_index, context_text, qa_text)

            # Save the current iteration number
            save_current_iteration(iteration)

if __name__ == "__main__":
    main()
