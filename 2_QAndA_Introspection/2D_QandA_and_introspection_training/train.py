# train.py

from agent import Agent
import random

# Import the q_category functions
from category_1_confidence import run_q_category_1
# from category_2_ import run_q_category_2
# from category_3_ import run_q_category_3
# from category_4_ import run_q_category_4
from category_5_self_recognition import run_q_category_5
# from category_6_ import run_q_category_6
from category_7_attention import run_q_category_7
# from category_8_ import run_q_category_8
# from category_9_ import run_q_category_9
from category_10_logits import run_q_category_10
# from category_11_ import run_q_category_11

def main():
    # Initialize the agent
    agent = Agent()

    # Main training loop
    num_iterations = 50
    q_categories = list(range(1, 12))  # Categories 1 to 11

    for iteration in range(num_iterations):
        print(f"Iteration {iteration+1}/{num_iterations}")
        q_category = random.choice(q_categories)
        if q_category == 1:
            loss, reward = run_q_category_1(agent)
        elif q_category == 2:
            loss, reward = run_q_category_2(agent)
        elif q_category == 3:
            loss, reward = run_q_category_3(agent)
        elif q_category == 4:
            loss, reward = run_q_category_4(agent)
        elif q_category == 5:
            loss, reward = run_q_category_5(agent)
        elif q_category == 6:
            loss, reward = run_q_category_6(agent)
        elif q_category == 7:
            loss, reward = run_q_category_7(agent)
        elif q_category == 8:
            loss, reward = run_q_category_8(agent)
        elif q_category == 9:
            loss, reward = run_q_category_9(agent)
        elif q_category == 10:
            loss, reward = run_q_category_10(agent)
        elif q_category == 11:
            loss, reward = run_q_category_11(agent)
        else:
            print(f"Invalid q_category: {q_category}")
            continue
        print(f"Loss: {loss:.4f} | Reward: {reward:.4f}")

if __name__ == "__main__":
    main()
