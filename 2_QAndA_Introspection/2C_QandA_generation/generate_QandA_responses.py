import json
import requests
import random
import os
from datetime import datetime
from datasets import load_dataset
from create_agents_if_not_exist import create_or_load_agents, print_agents

# OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')

# Get current date in yy_mm_dd format
current_date = datetime.now().strftime('%y_%m_%d')

def load_agents():
    agents = create_or_load_agents()
    return agents

def get_agent_descriptions(agents, agents_to_print):
    agent_descriptions = ''
    if agents_to_print:
        return print_agents(agents_to_print, agents)
        for agent_name in agents_to_print:
            continue
            properties = agents.get(agent_name, {})
            description = f"Agent [{agent_name}]: "
            property_descriptions = []
            for prop, value in properties.items():
                property_descriptions.append(f"{prop}: {value}")
            if property_descriptions:
                description += ', '.join(property_descriptions)
            else:
                description += 'No additional details.'
            agent_descriptions += description + "\n"
    else:
        agent_descriptions = "No agents are involved in this story.\n"
    return agent_descriptions

def load_glaive_dataset():
    ds = load_dataset("glaiveai/glaive-code-assistant")
    return ds

def generate_story_prompt(agents_to_print, special_agent, agents, dataset_row):
    if "python" not in (dataset_row['question'] + dataset_row['answer']).lower():
        return None
    # Get agent descriptions
    agent_descriptions = get_agent_descriptions(agents, agents_to_print)

    # Process the dataset row
    question_answer = f"Q: {dataset_row['question']}\nA: {dataset_row['answer']}"

    # Build the prompt
    prompt = f"""Your task is to provide context, and then a series of questions about that context. The topic will be about programming in Python.

The context you provide should pertain to the following question and its answer in Python:

START QUESTION AND ANSWER
{question_answer}
END QUESTION AND ANSWER

In your response, be sure to keep the example simple and clear, and remain in third person, referring to any actors in the context as "Agent [ALL_CAPS_NAME]".

Ensure the context you provide pertains to the question and answer above, in a simplified form.



Here is an example of good context and questions, although you should ensure the context you provide pertains to the question and answer above.

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

Be sure to refer to the following agents in your context:

{agent_descriptions}

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

def save_prompts_to_file(prompt, filename=f"prompts_{current_date}.txt"):
    with open(filename, "a") as file:
        file.write(prompt)
        file.write("<|SEPARATOR_OF_PAGES|>\n")

def save_responses_to_file(response, filename=f"responses_{current_date}.txt"):
    with open(filename, "a") as file:
        file.write(json.dumps(response))
        file.write("<|SEPARATOR_OF_PAGES|>\n")

def main():
    agents = load_agents()
    names_list = list(agents.keys())
    special_agent = names_list[0]

    ds = load_glaive_dataset()

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

        # Select a random row from the dataset
        random_index = random.randint(0, len(ds['train']) - 1)
        dataset_row = ds['train'][random_index]

        # Generate the prompt
        prompt = generate_story_prompt(agents_to_print, special_agent, agents, dataset_row)
        if prompt is None:
            continue

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

if __name__ == "__main__":
    main()
