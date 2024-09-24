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
    {"description": "Accurately talk about agents, including itself, in long form, and characterize them", "type": ["textbook learning"]}
]

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

def generate_prompt(agents_to_print, agents, group, dataset_row):
    if "python" not in (dataset_row['question'] + dataset_row['answer']).lower():
        return None
    # Get agent descriptions
    agent_descriptions = get_agent_descriptions(agents, agents_to_print)
    #prompt = "First, summarize the relevant information in the following Q&A in 1-3 sentences. Begin this answer with <|SUMMARY|> and end it with <|END SUMMARY|>."
    # Process the dataset row
    prompt = f"START Q&A WITH PYTHON CONTENT\nQ: {dataset_row['question']}\nA: {dataset_row['answer']}\nEND Q&A WITH PYTHON CONTENT\n\n\n"

    if agents_to_print:
        if len(agents_to_print) == 1:
            agent_word = "agent"
            extra_content_if_agents = " refer to agents as described,"
        else:
            agent_word = "agents"
            extra_content_if_agents = ""

        prompt += f"First, create some context for a Q&A, using the following {agent_word} as a character to refer to in third person in the context. Ensure the context and questions relate to the python content above. Do not explicitly list aspects of the agent in the context or ask about it in the questions. Ensure all specific functions and code needed to answer the question is included in the context.\n"
        prompt += f"BEGIN LIST OF AGENTS \n{agent_descriptions}\nEND LIST OF AGENTS\n\n"
        prompt += "Begin this answer with <|CONTEXT|> and end it with <|END CONTEXT|>.\n\n"
        prompt += f"Second, provide questions and answers relating to the context and background knowledge of a beginner agent.\n"
    else:
        prompt += "Second, create a series of questions and answers, also in the third person, using a beginner agent's background knowledge in Python. Do not mention any specific agents or use any names in your response.\n\n"

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

    # Determine if at least one 'quizzing' type question will be asked
    #quizzing_present = any("quizzing" in cat["type"] for cat in group)
    #if quizzing_present:
    #    prompt += "The 'quizzing' answers must have only one correct answer of at most 3 words.\n\n"
    prompt += f"When providing questions and answers, ensure they demonstrate the following relevant skills for the agent,{extra_content_if_agents} and teach basic information from the Q&A WITH PYTHON CONTENT.\n\n"

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

def call_openai_api(prompt):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    data = {
        "model": "gpt-4o-2024-08-06",
        "messages": [
            {"role": "system", "content": "You are an expert educator that teaches beginner agents language and programming knowledge."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1500
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

def main():
    agents = load_agents()
    names_list = list(agents.keys())
    special_agent = names_list[0]
    ds = load_glaive_dataset()
    rows_used = []
    for iteration in range(50):
        # Select a random row from the dataset
        random_index = random.randint(0, len(ds['train']) - 1)
        rows_used = rows_used.append(random_index)
        dataset_row = ds['train'][random_index]

        # Select number of agents based on weighted probabilities
        num_agents = random.choices([0, 1, 2, 3, 4], weights=[0.2, 0.25, 0.35, 0.15, 0.05])[0]

        if num_agents == 0:
            agents_to_print = []
        else:
            # Select unique agents excluding the special agent
            agents_to_print = random.sample([name for name in names_list if name != special_agent], num_agents)

            # 50% chance to include the special agent if any agents are being printed
            if random.random() < 0.5 and special_agent not in agents_to_print:
                if agents_to_print:
                    agents_to_print[0] = special_agent
                else:
                    agents_to_print.append(special_agent)

        #agent_descriptions = get_agent_descriptions(agents, agents_to_print)

        # Shuffle and group categories
        groups = shuffle_and_group_categories(categories)

        for group in groups:
            # Generate the prompt
            prompt = generate_prompt(agents_to_print, agents, group, dataset_row)
            print("prompt")
            print(prompt)
            print("\n\n\n\n\n\n\n\n")
        quit()

    # Call the GPT-4o model
    response = call_openai_api(prompt)

    # Print the response
    print("Response:")
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    main()
