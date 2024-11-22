# q_category_1.py

import random
from shared_utils import (
    tokenize_input,
    generate_model_answer,
    extract_confidence,
    check_answer_correctness,
    compute_loss_and_reward,
    perform_backward_pass
)

# List of programming concepts for q_category_1
programming_concepts = [
    "variables", "loops", "functions", "classes", "inheritance", "recursion",
    "data structures", "sorting algorithms", "search algorithms", "file I/O",
    "exception handling", "concurrency", "network programming", "database connectivity",
    "memory management", "object-oriented programming", "functional programming",
    "lambda expressions", "list comprehensions", "generators", "decorators",
    "regular expressions", "unit testing", "debugging", "design patterns"
]

def generate_question_and_answer(concept):
    # Placeholder function to generate question-answer pairs
    qa_pairs = {
        "variables": ("What is a variable in programming?", "A variable is a storage location paired with a symbolic name that contains some known or unknown quantity of information referred to as a value."),
        "loops": ("What is a loop in programming?", "A loop is a sequence of instructions that is continually repeated until a certain condition is reached."),
        "recursion": ("What is recursion in programming?", "Recursion is a technique where a function calls itself to solve smaller instances of a problem."),
        # Add more predefined questions and answers
    }
    if concept in qa_pairs:
        return qa_pairs[concept]
    else:
        # Default question and answer
        return (f"Explain the concept of {concept} in programming.", f"This is a placeholder answer for the concept of {concept}.")

def run_q_category_1(agent):
    # q_category_1 implementation
    concept = random.choice(programming_concepts)
    question, correct_answer = generate_question_and_answer(concept)
    input_text = f"Does agent [T5NAME] know the answer to question {question}?"
    input_ids = tokenize_input(input_text, agent.tokenizer, agent.device)
    # Get the model's answer
    model_answer = generate_model_answer(agent.model, agent.tokenizer, input_ids)
    agent_answer, confidence = extract_confidence(model_answer)
    is_correct = check_answer_correctness(agent_answer, correct_answer)
    # Compute loss and reward
    labels = agent.tokenizer.encode(correct_answer, return_tensors='pt').to(agent.device)
    adjusted_loss, reward = compute_loss_and_reward(agent.model, input_ids, labels, is_correct, confidence)
    # Perform backward pass
    perform_backward_pass(agent.optimizer, adjusted_loss)
    print(f"q_category_1 | Is Correct: {is_correct} | Confidence: {confidence} | Reward: {reward}")
    return adjusted_loss.item(), reward
