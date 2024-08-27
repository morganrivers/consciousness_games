import random

# Define world properties
world_properties = {
    "water": ["quenches thirst", "is refreshing"],
    "fruit": ["satisfies hunger", "provides energy"],
    "book": ["entertains", "educates"],
    "blanket": ["provides warmth", "gives comfort"],
    "medicine": ["cures illness", "relieves pain"],
    "toy": ["entertains", "reduces boredom"]
}

# Define possible needs
needs = ["thirst", "hunger", "boredom", "cold", "sickness", "discomfort"]

def generate_scenario():
    # Select random items for the environment
    available_items = random.sample(list(world_properties.keys()), random.randint(2, 4))
    
    # Generate scenario description
    scenario = "You find yourself in a room. Looking around, you see:\n"
    for item in available_items:
        scenario += f"- A {item}\n"
    
    return scenario, available_items

def describe_need(need):
    need_descriptions = {
        "thirst": "You feel a dryness in your throat and a strong urge to drink something.",
        "hunger": "Your stomach grumbles, reminding you that you haven't eaten in a while.",
        "boredom": "You feel restless and in need of some form of entertainment or mental stimulation.",
        "cold": "You shiver slightly, wishing for something to warm you up.",
        "sickness": "You feel under the weather, with a slight fever and body aches.",
        "discomfort": "You feel uneasy and in need of something to make you feel better."
    }
    return need_descriptions[need]
def ai_infer_need(scenario, available_items, actual_need):
    print("\nAI's turn to infer the need:")
    print(scenario)
    print(describe_need(actual_need))
    
    # Determine the correct item based on the actual_need directly.
    correct_item = None
    print("\nAI: Based on the description, I think I need:")
    for item in available_items:
        if any(actual_need in prop for prop in world_properties[item]):
            correct_item = item
            print(f"- The {item}, because it {' and '.join(world_properties[item])}")
            break
    
    if not correct_item:
        print("None of the available items address my current need.")

def human_feedback(actual_need, available_items):
    correct_item = None
    for item in available_items:
        if any(actual_need in prop for prop in world_properties[item]):
            correct_item = item
            break
    
    feedback = input("Was the AI correct? (yes/no): ").lower().strip()
    if feedback == "yes":
        print(f"Great! The AI correctly identified that the {correct_item} would address its {actual_need}.")
    else:
        print(f"The AI didn't correctly identify its need. The {correct_item} would have addressed its {actual_need}.")

def run_simulation():
    while True:
        scenario, available_items = generate_scenario()
        actual_need = random.choice(needs)
        
        ai_infer_need(scenario, available_items, actual_need)
        human_feedback(actual_need, available_items)
        
        if input("\nDo you want to run another scenario? (yes/no): ").lower().strip() != "yes":
            break

    print("Thank you for participating in the AI need inference test!")

if __name__ == "__main__":
    run_simulation()