import random
import string

# Generate 10 unique 7-letter all-caps random names
names_set = set()
while len(names_set) < 10:
    name = ''.join(random.choices(string.ascii_uppercase, k=7))
    names_set.add(name)
names_list = list(names_set)

# The first agent is special
special_agent = names_list[0]

# Loop over 50 iterations
for _ in range(50):
    # Number of agents to print this iteration.
    # 0 → 20%, 1 → 25%, 2 → 35%, 3 → 15%, 4 → 5%  (# agents -> % of the time that many agents)
    num_agents = random.choices([0, 1, 2, 3, 4], weights=[0.2, 0.25, 0.35, 0.15, 0.05])[0]

    if num_agents == 0:
        print("\nno agents")
    else:
        print()
        # Select unique agents for this iteration
        agents_to_print = random.sample([name for name in names_list if name != special_agent], num_agents)

        # 50% chance to include the special agent if any agents are being printed
        if num_agents > 0 and random.random() < 0.5:
            # Add the special agent to the list, ensuring unique selection
            agents_to_print[0] = special_agent

        for agent_name in agents_to_print:
            if agent_name == special_agent:
                print(f"Agent {agent_name} special")
            else:
                print(f"Agent {agent_name}")
