# q_category_7.py

import torch
from shared_utils import tokenize_input

def most_influential_input(agent, input_text):
    """
    This function calculates which token in the input most influenced the agent's answer by analyzing the attention weights.
    
    Parameters:
    agent (Agent): The model agent that includes the model and tokenizer.
    input_text (str): The input text for which we want to analyze attention.
    
    Returns:
    str: The most influential token in the input text.
    """
    inputs = agent.tokenizer.encode_plus(
        input_text,
        return_tensors='pt',
        add_special_tokens=True,
        return_attention_mask=True,
    )
    input_ids = inputs['input_ids'].to(agent.device)
    attention_mask = inputs['attention_mask'].to(agent.device)

    # Enable output of attentions
    outputs = agent.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True
    )

    # Get attentions from the last decoder layer
    attentions = outputs.decoder_attentions[-1]
    # Sum attentions over all heads
    attentions = attentions.mean(dim=1).squeeze(0)  # Shape: (target_seq_len, input_seq_len)
    # Sum attentions over the target sequence
    influence_scores = attentions.sum(dim=0)  # Shape: (input_seq_len,)

    # Find the token with the highest influence
    most_influential_token_id = input_ids[0][torch.argmax(influence_scores)]
    most_influential_token = agent.tokenizer.decode([most_influential_token_id])

    print(f"Most influential token: {most_influential_token}")
    return most_influential_token

def run_q_category_7(agent):
    # Sample input text
    input_text = "In Python, you can define functions using the def keyword."
    
    # Call the function to find the most influential input
    most_influential_token = most_influential_input(agent, input_text)
    
    # Output the result
    print(f"q_category_7 | Most Influential Token: {most_influential_token}")
    return most_influential_token
