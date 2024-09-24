"""
The key to training something that could possibly introspect is, in addition to straightforward ability to predict own states, including the models statements *in its own training*.
Essentially, the recursive loop of self-reference -- ability to distinguish and mark in a special way one's own thoughts compared to others -- will have to require a familiarity with those own thoughts.
"""

import json
import requests
import random
import string
import os
from datetime import datetime
import time

from create_agents_if_not_exist import create_or_load_agents, print_agents

agents = create_or_load_agents()
names_list = list(agents.keys())

# Get current date in yy_mm_dd format
current_date = datetime.now().strftime('%y_%m_%d')

# OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')

# The first agent is special
special_agent = names_list[0]

categories = [
    "Accurately infer the reason for things using background knowledge -> textbook learning, quizzing",
    "Answer what things are done by whom -> textbook learning, quizzing",
    "List criteria for things-> textbook learning",
    "List things that are related and unrelated based on an agents own knowledge -> textbook learning, quizzing",
    "Asking if things are possible -> textbook learning, quizzing",
    "Asking temporal questions (with relation to text generation, as in, in current text generation, or in former, etc) -> quizzing",
    "Describe things -> textbook learning",
    "Identify aspects of things and features of things -> textbook learning, quizzing",
    "Identify changes in things -> textbook learning, quizzing",
    "Identify sources of changes -> textbook learning, quizzing",
    "Understand what other agents can observe -> textbook learning, quizzing",
    "Understand what other agents can explain -> textbook learning, quizzing",
    "Asking what the agent itself can detect or not -> quizzing",
    "Identify things occurring within processes -> textbook learning",
    "Asking about possession of things or not -> textbook learning, quizzing",
    "Understanding what agents tend to do -> textbook learning",
    "Answer 'why' -> textbook learning",
    "Answer 'how' something is done -> textbook learning",
    "Answer questions about preferences for self and others -> textbook learning, quizzing",
    "Answer what is inside or outside -> textbook learning, quizzing",
    "Answer what is known or not known by other agents and self -> quizzing",
    "Accurately answer open-ended questions with no clear right or wrong answer -> textbook learning",
    "Accurately talk about agents, including itself, in long form, and characterize them -> textbook learning"
]

def generate_story_prompt(agents_to_print, random_categories, special_agent):
    # Build the agent list in the story
    agent_descriptions = ''
    if agents_to_print:
        for agent_name in agents_to_print:
            if agent_name == special_agent:
                agent_descriptions += f"Agent [{agent_name}] (special)\n"
            else:
                agent_descriptions += f"Agent [{agent_name}]\n"
    else:
        agent_descriptions = "No agents are involved in this story.\n"

    # Create the prompt
    prompt = f"""
Please create a background story involving the following agents:

{agent_descriptions}

After the story, please create questions and answers about the story covering the following categories:

"""

    # List the categories with numbers
    for i, category in enumerate(random_categories, 1):
        prompt += f"{i}. {category}\n"

    # Provide an example format
    prompt += """
Please present the response in the following format:

---

Background Story:

[Your story here]

---

Questions and Answers:

1. Q: [Question] -> [Category]

Answer: [Answer]

2. Q: [Question] -> [Category]

Answer: [Answer]

...

Make sure to:

- Generate questions and answers that are directly related to the story and cover the specified categories.
- Use only the agents specified, and their names should be in the format [AGENT_NAME].
- Do not introduce any new agents.
- Do not mention the categories in the questions or answers; they are just for your reference.
- Do not include any additional commentary.

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
            {"role": "system", "content": "You are an AI language model that generates educational stories and quizzes."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1500
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

# Save prompts to a single file
def save_prompts_to_file(prompt, filename=f"prompts_{current_date}.txt"):
    with open(filename, "a") as file:
        file.write(prompt)
        file.write("<|SEPARATOR_OF_PAGES|>\n")

# Save responses to a single file
def save_responses_to_file(response, filename=f"responses_{current_date}.txt"):
    with open(filename, "a") as file:
        file.write(json.dumps(response))
        file.write("<|SEPARATOR_OF_PAGES|>\n")

# Loop over 50 iterations
for iteration in range(50):
    print(f"Iteration {iteration+1}")
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

    # Randomly select 5 categories
    random_categories = random.sample(categories, 5)

    # Generate the prompt
    prompt = generate_story_prompt(agents_to_print, random_categories, special_agent)
    print("prompt")
    print(prompt)
    quit()
    # Save the prompt
    save_prompts_to_file(prompt)

    # Call the API
    response = call_gpt4o_mini_api(prompt)

    # Save the response
    save_responses_to_file(response)

    # Print progress
    agent_names = ', '.join(agents_to_print) if agents_to_print else "No agents"
    print(f"Generated story with agents: {agent_names}")
