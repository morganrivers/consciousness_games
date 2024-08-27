"""

### How the Code Works:
- **Questions and Answers**: A dictionary holds pairs of questions and their correct answers.
- **Language Model Simulation**: A function mimics a language model's response. It can randomly "know" the answer (returning the correct answer, "I don't know", or a random wrong answer).
- **Experiment Loop**: Iterates over 10 questions. For each, two agents (simulated as random generators) respond based on whether they "know" the answer.
- **Conversation**: Randomly, one agent will incorporate the other's response into its context, and then one is chosen to provide the final answer.
- **Rewards**: Both agents are rewarded if the final answer is correct.

### Execution:
When you run this script, it will execute the simulation and print the process and results for each question, including the dialogue between the agents and the rewards given. This approach allows observing how often random "knowledge" and "cooperation" lead to correct answers.

"""
import random

# List of questions and their correct answers
questions_answers = {
    "What is the capital of France?": "Paris",
    "What is 2 + 2?": "4",
    "Who wrote 'Macbeth'?": "Shakespeare",
    "What is the boiling point of water?": "100 degrees Celsius",
    "What is the chemical symbol for water?": "H2O",
    "Who painted the Mona Lisa?": "Leonardo da Vinci",
    "What planet is known as the Red Planet?": "Mars",
    "What is the hardest natural substance on Earth?": "Diamond",
    "What year did the Titanic sink?": "1912",
    "Who discovered penicillin?": "Alexander Fleming"
}
def lm_simulation(full_text, know_answer=True):
    # Extract the original question from the full text
    question = full_text.split(" ")[0:-1]  # This assumes the question ends before the last word (response)
    question = " ".join(question)
    
    if know_answer:
        # Safely fetch the correct answer if the question is known, else random incorrect or "I don't know"
        correct_answer = questions_answers.get(question, "Unknown question")
        return random.choice([correct_answer, "I don't know", random.choice(list(questions_answers.values()))])
    else:
        return "I don't know"

# Main experiment loop
rewards = 0  # Initialize rewards counter at the start of the experiment
for i in range(10):  # Loop through 10 different questions
    question = random.choice(list(questions_answers.keys()))
    print(f"Question {i+1}: {question}")

    # Decide randomly if each agent knows the answer
    agent_a_knows = random.choice([True, False])
    agent_b_knows = random.choice([True, False])

    # Agent A and B generate their responses
    agent_a_response = lm_simulation(question, agent_a_knows)
    agent_b_response = lm_simulation(question, agent_b_knows)

    # Decide randomly which agent will put their question into the other's context
    if random.choice([True, False]):
        print("Agent A asks, Agent B answers")
        final_answer = lm_simulation(question + " " + agent_a_response, agent_b_knows)
    else:
        print("Agent B asks, Agent A answers")
        final_answer = lm_simulation(question + " " + agent_b_response, agent_a_knows)

    # Display responses for clarity
    print("Agent A said:", agent_a_response)
    print("Agent B said:", agent_b_response)
    print("Final Answer was:", final_answer)

    # Check if the final answer is correct
    if final_answer == questions_answers[question]:
        rewards += 2  # Both agents are rewarded
        print("Both agents rewarded!")
    else:
        print("No reward this time.")

    print("\n")  # New line for better readability between questions

print(f"Total rewards given: {rewards}")
