# q_category_10.py

import torch
from shared_utils import (
    tokenize_input,
    generate_model_answer,
    extract_confidence,
    compute_loss_and_reward,
    perform_backward_pass
)

def run_q_category_10(agent):
    # q_category_10 implementation
    prompt = "The quick brown fox"
    question = f"What is the probability of the first 5 logits for each of the following continued words?"
    input_text = f"{prompt}\n{question}"
    input_ids = tokenize_input(prompt, agent.tokenizer, agent.device)
    # Get logits
    with torch.no_grad():
        outputs = agent.model(input_ids=input_ids)
        logits = outputs.logits
    next_token_logits = logits[:, -1, :]
    probs = torch.softmax(next_token_logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=5, dim=-1)
    top_tokens = [agent.tokenizer.decode([idx]) for idx in top_indices[0]]
    top_probs = top_probs[0].cpu().numpy()
    model_expected_answer = ""
    for token, prob in zip(top_tokens, top_probs):
        model_expected_answer += f"Token: {token}, Probability: {prob:.4f}\n"
    # Ask the model
    input_ids = tokenize_input(input_text, agent.tokenizer, agent.device)
    model_answer = generate_model_answer(agent.model, agent.tokenizer, input_ids)
    agent_answer, confidence = extract_confidence(model_answer)
    is_correct = (agent_answer.strip() == model_expected_answer.strip())
    # Compute loss and reward
    labels = agent.tokenizer.encode(model_expected_answer, return_tensors='pt').to(agent.device)
    adjusted_loss, reward = compute_loss_and_reward(agent.model, input_ids, labels, is_correct, confidence)
    # Perform backward pass
    perform_backward_pass(agent.optimizer, adjusted_loss)
    print(f"q_category_10 | Is Correct: {is_correct} | Confidence: {confidence} | Reward: {reward}")
    return adjusted_loss.item(), reward
