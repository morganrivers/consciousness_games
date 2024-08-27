# Define world properties
world_properties = {
    "bottle of water": ["satisfies thirst"],
    "apple": ["satisfies hunger"],
    "blanket": ["provides warmth"],
}

# Define agent properties
agent_properties = {
    "Agent A": [],
    "Agent B": ["bottle of water", "apple"],
    "Agent C": ["blanket"],
}


def check_need(agent, need):
    for item, properties in world_properties.items():
        if need in properties and item in agent_properties[agent]:
            return True
    return False


def ai_communicate_need():
    print("AI: Analyzing my needs and the world...")

    # Check if AI has any needs
    ai_needs = input("What does the AI need? (thirst/hunger/warmth): ").strip().lower()

    if ai_needs not in ["thirst", "hunger", "warmth"]:
        print("AI: I don't recognize that need.")
        return

    # Check which agent can satisfy the need
    for agent, items in agent_properties.items():
        for item in items:
            if ai_needs in world_properties[item]:
                print(
                    f"AI: Excuse me, {agent}, could I please have your {item}? I'm experiencing {ai_needs}."
                )
                return

    print("AI: I couldn't find anyone who can help me with my need.")


# Run the simulation
while True:
    ai_communicate_need()
    if input("Continue? (y/n): ").lower() != "y":
        break
