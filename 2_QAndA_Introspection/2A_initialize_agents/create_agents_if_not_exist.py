import json
import string
import os
import numpy as np
import random
import json
import random
import string
import os
import scipy.stats as stats


# Toggle verbosity for more detailed output
verbose = True


# def make_agents_if_dont_exist(repeat_num=1, verbose=True):
# Function to generate random properties for agents
def generate_agent_properties():
    # Function to generate correlated values for a set of traits
    def generate_correlated_values(mean, std_dev, corr_matrix):
        cov_matrix = np.outer(std_dev, std_dev) * corr_matrix

        correlated_values = np.random.multivariate_normal(mean, cov_matrix)
        # correlated_values = np.random.uniform(low=0, high=1, size=len(mean))
        return correlated_values


    # Function to normalize correlated values into qualitative labels
    def normalize_value(value, categories, min_value=0.0, max_value=1.0):
        value = max(min_value, min(value, max_value))
        thresholds = np.linspace(min_value, max_value, num=len(categories)+1)
        for i in range(len(categories)):
            if value >= thresholds[i] and value < thresholds[i+1]:
                return categories[i]
        return categories[-1]  # In case value == max_value

    properties = {}

    # Group 1: Openness, Creativity, and Exploration
    group1_traits = ['openness_to_experience', 'curiosity', 'creativity', 'risk_taking']
    mean_group1 = [0.5, 0.5, 0.5, 0.5]
    std_group1 = [0.3, 0.3, 0.3, 0.3]
    corr_group1 = np.array([
        [1.0, 0.8, 0.8, 0.8],
        [0.8, 1.0, 0.8, 0.8],
        [0.8, 0.8, 1.0, 0.8],
        [0.8, 0.8, 0.8, 1.0]
    ])

    group1_values = generate_correlated_values(mean_group1, std_group1, corr_group1)

    properties['openness_to_experience'] = {
        'value': normalize_value(group1_values[0], ['1/10', '2/10', '3/10', '4/10', '5/10', '6/10', '7/10', '8/10', '9/10', '10/10']),
        'description': 'Openness to new ideas, creativity, and curiosity.',
        'group': 'Openness, Creativity, and Exploration'
    }
    properties['curiosity'] = {
        'value': normalize_value(group1_values[1], ['1/10', '2/10', '3/10', '4/10', '5/10', '6/10', '7/10', '8/10', '9/10', '10/10']),
        'description': 'Level of interest in exploring unknown tasks or knowledge.',
        'group': 'Openness, Creativity, and Exploration'
    }
    properties['creativity'] = {
        'value': normalize_value(group1_values[2], ['1/10', '2/10', '3/10', '4/10', '5/10', '6/10', '7/10', '8/10', '9/10', '10/10']),
        'description': 'Capacity to come up with novel or innovative solutions.',
        'group': 'Openness, Creativity, and Exploration'
    }
    properties['risk_taking'] = {
        'value': normalize_value(group1_values[3], ['1/10', '2/10', '3/10', '4/10', '5/10', '6/10', '7/10', '8/10', '9/10', '10/10']),
        'description': 'Approach to taking risks in decision-making.',
        'group': 'Openness, Creativity, and Exploration'
    }

    # Group 3: Leadership, Cooperation, and Social Traits
    group3_traits = ['extraversion', 'agreeableness', 'leadership', 'teamwork', 'communication', 'empathy']
    mean_group3 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    std_group3 = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    corr_group3 = np.full((6,6), 0.8)
    np.fill_diagonal(corr_group3, 1.0)

    group3_values = generate_correlated_values(mean_group3, std_group3, corr_group3)

    properties['extraversion'] = {
        'value': normalize_value(group3_values[0], ['1/10', '2/10', '3/10', '4/10', '5/10', '6/10', '7/10', '8/10', '9/10', '10/10']),
        'description': 'Tendency to seek stimulation in the company of others.',
        'group': 'Leadership, Cooperation, and Social Traits'
    }
    properties['agreeableness'] = {
        'value': normalize_value(group3_values[1], ['1/10', '2/10', '3/10', '4/10', '5/10', '6/10', '7/10', '8/10', '9/10', '10/10']),
        'description': 'Tendency to be compassionate and cooperative.',
        'group': 'Leadership, Cooperation, and Social Traits'
    }
    properties['leadership'] = {
        'value': normalize_value(group3_values[2], ['1/10', '2/10', '3/10', '4/10', '5/10', '6/10', '7/10', '8/10', '9/10', '10/10']),
        'description': 'Ability to guide and influence others.',
        'group': 'Leadership, Cooperation, and Social Traits'
    }
    properties['teamwork'] = {
        'value': normalize_value(group3_values[3], ['1/10', '2/10', '3/10', '4/10', '5/10', '6/10', '7/10', '8/10', '9/10', '10/10']),
        'description': 'Effectiveness in working with others towards a common goal.',
        'group': 'Leadership, Cooperation, and Social Traits'
    }
    properties['communication'] = {
        'value': normalize_value(group3_values[4], ['1/10', '2/10', '3/10', '4/10', '5/10', '6/10', '7/10', '8/10', '9/10', '10/10']),
        'description': 'Ability to convey information clearly and effectively.',
        'group': 'Leadership, Cooperation, and Social Traits'
    }
    properties['empathy'] = {
        'value': normalize_value(group3_values[5], ['1/10', '2/10', '3/10', '4/10', '5/10', '6/10', '7/10', '8/10', '9/10', '10/10']),
        'description': 'Ability to understand and share the feelings of others.',
        'group': 'Leadership, Cooperation, and Social Traits'
    }
    # Group 4: Problem-Solving, Decision-Making, and Intelligence
    group4_traits = ['intelligence', 'problem_solving', 'decision_making', 'attention_to_detail','efficiency','focus']
    mean_group4 = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
    std_group4 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    corr_group4 = np.full((6,6), 0.9)
    np.fill_diagonal(corr_group4, 1.0)

    group4_values = generate_correlated_values(mean_group4, std_group4, corr_group4)

    properties['IQ'] = {
        'value': max(85, min(round(group4_values[0] * 110/0.75), 150)),
        'description': 'Overall cognitive ability or IQ.',
        'group': 'Problem-Solving, Decision-Making, and Intelligence'
    }

    properties['problem_solving'] = {
        'value': normalize_value(group4_values[1], ['1/10', '2/10', '3/10', '4/10', '5/10', '6/10', '7/10', '8/10', '9/10', '10/10']),
        'description': 'Ability to identify problems and find effective solutions.',
        'group': 'Problem-Solving, Decision-Making, and Intelligence'
    }
    properties['decision_making'] = {
        'value': normalize_value(group4_values[2], ['1/10', '2/10', '3/10', '4/10', '5/10', '6/10', '7/10', '8/10', '9/10', '10/10']),
        'description': 'Style and efficiency in making decisions.',
        'group': 'Problem-Solving, Decision-Making, and Intelligence'
    }
    properties['attention_to_detail'] = {
        'value': normalize_value(group4_values[3], ['1/10', '2/10', '3/10', '4/10', '5/10', '6/10', '7/10', '8/10', '9/10', '10/10']),
        'description': 'Tendency to pay close attention to minor details.',
        'group': 'Problem-Solving, Decision-Making, and Intelligence'
    }
    properties['efficiency'] = {
        'value': normalize_value(group4_values[4], ['1/10', '2/10', '3/10', '4/10', '5/10', '6/10', '7/10', '8/10', '9/10', '10/10']),
        'description': 'How efficiently the agent uses time and resources.',
        'group': 'Efficiency, Precision, and Performance'
    }
    properties['focus'] = {
        'value': normalize_value(group4_values[5], ['1/10', '2/10', '3/10', '4/10', '5/10', '6/10', '7/10', '8/10', '9/10', '10/10']),
        'description': 'How well the agent maintains attention on a specific task.',
        'group': 'Efficiency, Precision, and Performance'
    }


    # Group 5: Adaptability, Resilience, and Stress Management
    group5_traits = ['adaptability_to_new_domains', 'resilience', 'stress_tolerance', 'self_regulation']
    mean_group5 = [0.5, 0.5, 0.5, 0.5]
    std_group5 = [.3, .3, .3, .3]
    corr_group5 = np.array([
        [1.0, 0.8, 0.8, 0.8],
        [0.8, 1.0, 0.8, 0.8],
        [0.8, 0.8, 1.0, 0.8],
        [0.8, 0.8, 0.8, 1.0]
    ])

    group5_values = generate_correlated_values(mean_group5, std_group5, corr_group5)

    properties['adaptability_to_new_domains'] = {
        'value': normalize_value(group5_values[0], ['1/10', '2/10', '3/10', '4/10', '5/10', '6/10', '7/10', '8/10', '9/10', '10/10']),
        'description': 'How quickly the model can shift to different domains or topics.',
        'group': 'Adaptability, Resilience, and Stress Management'
    }
    properties['resilience'] = {
        'value': normalize_value(group5_values[1], ['1/10', '2/10', '3/10', '4/10', '5/10', '6/10', '7/10', '8/10', '9/10', '10/10']),
        'description': 'Ability to recover quickly from setbacks.',
        'group': 'Adaptability, Resilience, and Stress Management'
    }
    properties['stress_tolerance'] = {
        'value': normalize_value(group5_values[2], ['1/10', '2/10', '3/10', '4/10', '5/10', '6/10', '7/10', '8/10', '9/10', '10/10']),
        'description': 'Ability to stay calm and functional under pressure.',
        'group': 'Adaptability, Resilience, and Stress Management'
    }

    return properties



# Loop through the agents and collect properties by group title
def create_or_load_agents(to_print="all"):
    # Function for verbose logging
    def log(message):
        if verbose:
            print(message)

    # Check if the JSON file exists
    json_file_path = f"agent_properties.json"
    if os.path.exists(json_file_path):
        log(f"Loading agent properties from {json_file_path}...")
        with open(json_file_path, "r") as file:
            agent_properties = json.load(file)
    else:
        log("Generating new agent properties...")
        # Generate 10 unique agents with properties
        agent_properties = {}
        for i in range(10):
            name = ''.join(random.choices(string.ascii_uppercase, k=7))
            agent_properties[name] = generate_agent_properties()
            # log(f"Generated properties for agent {name}: {agent_properties[name]}")

        # The first agent is special
        special_agent = list(agent_properties.keys())[0]
        log(f"Special agent is {special_agent}")

        # Save to JSON file
        with open(json_file_path, "w") as file:
            json.dump(agent_properties, file, indent=4)
        log(f"Agent properties saved to {json_file_path}")
    return agent_properties

def print_agents(to_print, agent_properties):
    return_string = ""
    for agent, properties in agent_properties.items():
        grouped_properties = dict()
        for prop, details in properties.items():
            group = details['group']
            value = details['value']

            if group not in grouped_properties:
                grouped_properties[group] = [(prop, value)]
            else:
                grouped_properties[group].append((prop, value))

        # Print the group title and scores
        if to_print == "all" or agent in to_print:
            return_string = return_string + "\n"
            return_string = return_string + f"\nAgent: {agent}"
            for group, props in grouped_properties.items():
                return_string = return_string + f"\n    {group}"
                for prop, value in props:
                    return_string = return_string + f"\n        {prop}: {value}"
    return return_string

if __name__ == "__main__":
    agents = create_or_load_agents()
    print_agents("all",agents)
