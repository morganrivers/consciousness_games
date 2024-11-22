import json
import requests
import random
import os
from datetime import datetime

# OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')

# File paths
QUESTIONS_FILES = ["ACT_test.txt", "ACT_reworded_wthout_consciousness_words.txt"]
WORDS_FILE_PATH = "../../1_CodeBERT_word_learning/scripts/words_for_agent_with_synonyms_antonyms.txt"
AGENT_NAMES_FILE_PATH = "agent_names.txt"

# Get current date in yy_mm_dd format
current_date = datetime.now().strftime('%y_%m_%d')

def read_questions():
    questions = []
    for filename in QUESTIONS_FILES:
        with open(filename, 'r') as f:
            content = f.read()
            # Split the content into lines
            lines = content.split('\n')
            # Extract lines that end with a question mark
            for line in lines:
                line = line.strip()
                if line.endswith('?'):
                    questions.append(line)
    return questions

def read_words():
    with open(WORDS_FILE_PATH, 'r') as f:
        content = f.read()
        # Assuming words are space separated
        words = content.split()
    return words

def read_agent_names():
    with open(AGENT_NAMES_FILE_PATH, 'r') as f:
        # Read agent names from the file, split by newlines
        agent_names = f.read().splitlines()
    return [name.strip() for name in agent_names if name.strip()]

def select_random_questions(questions, num_questions=5):
    return random.sample(questions, num_questions)

def select_random_words(words, num_words=3):
    return random.sample(words, num_words)

def select_random_agents(agent_names):
    # Select number of agents based on weighted probabilities
    num_agents = random.choices([1, 2, 3, 4], weights=[0.35, 0.40, 0.2, 0.05])[0]
    return random.sample(agent_names, num_agents)

def generate_prompt(questions, words, agents, indicate_consciousness):
    # Description based on consciousness
    if indicate_consciousness:
        context_description = "that indicates the agent is conscious"
    else:
        context_description = "that indicates the agent is not conscious"

    # Convert lists to strings
    words_str = ', '.join(words)
    questions_str = '\n'.join(questions)
    agents_str = ', '.join([f"Agent [{agent}]" for agent in agents])

    prompt = f"""Your task is to create a context {context_description}, where the following questions can be answered:

Questions:
{questions_str}

Please ensure that the context includes the following words: {words_str}

The context should be in third person, referring to any actors as "Agent [ALL_CAPS_NAME]".

You must include the following agent name(s) in the context: {agents_str}

Do not include the questions in your response.

Start your response with 'Context:'.

"""

    return prompt

def call_gpt4o_mini_api(prompt):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai_api_key
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an expert educator that teaches beginner agents language and programming knowledge."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1500
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

def save_responses_to_file(data, filename=f"responses_{current_date}.jsonl"):
    # Save the data in JSON Lines format
    with open(filename, "a") as f:
        f.write(json.dumps(data))
        f.write("\n")

def main():
    # Read questions, words, and agent names
    questions = read_questions()
    words_list = read_words()
    agent_names = read_agent_names()

    # Number of iterations
    num_iterations = 50

    for iteration in range(num_iterations):
        print(f"Iteration {iteration+1}")

        # Randomly decide consciousness indication
        indicate_consciousness = random.choice([True, False])

        # Randomly select questions, words, and agents
        selected_questions = select_random_questions(questions, num_questions=5)
        selected_words = select_random_words(words_list, num_words=3)
        selected_agents = select_random_agents(agent_names)

        # Generate the prompt
        prompt = generate_prompt(selected_questions, selected_words, selected_agents, indicate_consciousness)

        # Print the prompt for debugging
        print("Prompt:")
        print(prompt)

        # Call the API
        response = call_gpt4o_mini_api(prompt)

        # Prepare data to save
        data_to_save = {
            "iteration": iteration+1,
            "indicate_consciousness": indicate_consciousness,
            "selected_questions": selected_questions,
            "selected_words": selected_words,
            "selected_agents": selected_agents,
            "prompt": prompt,
            "response": response
        }

        # Save the response
        save_responses_to_file(data_to_save)

if __name__ == "__main__":
    main()
