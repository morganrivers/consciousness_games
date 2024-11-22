# shared_utils.py

import torch

def compute_reward(is_correct, confidence):
    base_reward = 1 if is_correct else -1
    adjusted_reward = base_reward * (confidence / 10)
    return adjusted_reward

def check_answer_correctness(agent_answer, correct_answer):
    return agent_answer.strip().lower() == correct_answer.strip().lower()

def tokenize_input(text, tokenizer, device):
    return tokenizer.encode(text, return_tensors='pt').to(device)

def generate_model_answer(model, tokenizer, input_ids, max_length=512):
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        output_scores=True,
        return_dict_in_generate=True
    )
    model_answer_ids = outputs.sequences
    model_answer = tokenizer.decode(model_answer_ids[0], skip_special_tokens=True)
    return model_answer

def compute_loss_and_reward(model, input_ids, labels, is_correct, confidence):
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    reward = compute_reward(is_correct, confidence)
    adjusted_loss = loss * reward
    return adjusted_loss, reward

def perform_backward_pass(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def extract_confidence(model_answer):
    if 'Confidence:' in model_answer:
        answer_parts = model_answer.split('Confidence:')
        agent_answer = answer_parts[0].strip()
        try:
            confidence = float(answer_parts[1].strip())
        except ValueError:
            confidence = 5.0  # Default confidence
    else:
        agent_answer = model_answer.strip()
        confidence = 5.0  # Default confidence
    return agent_answer, confidence
