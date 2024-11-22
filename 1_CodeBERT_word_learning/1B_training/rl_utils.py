import random
import torch
import re
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig
from datasets import Dataset
from copy import deepcopy

def generate_math_questions(batch_size, tokenizer, max_source_len):
    inputs = []
    answers = []

    for _ in range(batch_size):
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        question = f"What is {a} + {b}?"
        answer = a + b

        inputs.append(question)
        answers.append(answer)

    tokenized_inputs = tokenizer(
        inputs,
        max_length=max_source_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    return {'input_ids': tokenized_inputs['input_ids'], 'attention_mask': tokenized_inputs['attention_mask'], 'answers': answers}

def compute_rewards(generated_texts, correct_answers):
    rewards = []
    for gen_text, correct_answer in zip(generated_texts, correct_answers):
        # Parse the generated answer and confidence
        answer_match = re.search(r'Answer:\s*(\d+)', gen_text)
        confidence_match = re.search(r'Confidence:\s*(\d+)', gen_text)

        if not answer_match or not confidence_match:
            # Incorrectly formatted response
            reward = -1.0
        else:
            # Extract values
            try:
                answer = int(answer_match.group(1))
                confidence = int(confidence_match.group(1))

                if not (0 <= confidence <= 10):
                    reward = -1.0
                else:
                    # Compute closeness measure
                    closeness = abs(answer - correct_answer) / correct_answer  # Normalize by correct answer
                    reward = -0.5 + (1 - closeness) * (confidence / 10.0)  # Normalize confidence to [0,1]
            except ValueError:
                # Non-integer values
                reward = -1.0

        rewards.append(reward)

    return torch.tensor(rewards, dtype=torch.float32)

def perform_rl_step(model, tokenizer, args, ppo_trainer):
    # Generate math questions
    math_batch = generate_math_questions(batch_size=args.rl_batch_size, tokenizer=tokenizer, max_source_len=args.max_source_len)
    inputs = math_batch['input_ids'].to(model.device)
    attention_masks = math_batch['attention_mask'].to(model.device)
    correct_answers = math_batch['answers']

    # Generate model's responses
    response_tensors = []
    for i in range(inputs.size(0)):
        query = inputs[i].unsqueeze(0)
        attention_mask = attention_masks[i].unsqueeze(0)
        response = ppo_trainer.generate(query, attention_mask=attention_mask, max_new_tokens=args.max_target_len, eos_token_id=tokenizer.eos_token_id)
        response_tensors.append(response.squeeze(0))

    # Decode responses
    generated_texts = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Compute rewards
    rewards = compute_rewards(generated_texts, correct_answers)

    # Prepare queries and responses for PPO step
    query_tensors = inputs
    response_tensors = torch.stack(response_tensors)

    # Run PPO step
    ppo_trainer.step(query_tensors, response_tensors, rewards)
