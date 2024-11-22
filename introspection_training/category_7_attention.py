import torch
from shared_utils import tokenize_input
from IPython import embed
from attention_utils import print_attention_on_tokens, get_attn_over_all_layers

# added import HTML for print_attention function
#from IPython.core.display import display, HTML

# def get_length_without_special_tokens(sentence):
#     length = 0
#     for i in sentence:
#         if i == 0:
#             break
#         else:
#             length += 1
#     return length

# def print_attention(input_ids_all, attentions_all, tokenizer):
#     for input_ids, attention in zip(input_ids_all, attentions_all):
#         html = []
#         len_input_ids = get_length_without_special_tokens(input_ids)
#         input_ids = input_ids[:len_input_ids]
#         attention = attention[:len_input_ids]
#         for input_id, attention_value in zip(input_ids, attention):
#             token = tokenizer.convert_ids_to_tokens(input_id)
#             attention_value = attention_value
#             html.append('<span style="background-color: rgb(255,255,0,{0})">{1}</span>'.format(10 * attention_value, token))
#         html_string = " ".join(html)
#         print(HTML(html_string))


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
        add_special_tokens=False,
        return_attention_mask=True,
    )
    input_ids = inputs['input_ids'].to(agent.device)

    # Generate a dummy empty tensor for decoder_input_ids for `model.generate`
    # input_ids will be used as `encoder_input_ids`
    decoder_input_ids = torch.ones(1, 1, dtype=torch.long)
    decoder_input_ids.fill_(agent.tokenizer.pad_token_id)
    decoder_input_ids = decoder_input_ids.to(agent.device)

    input_ids = inputs['input_ids'].to(agent.device)

    # attention_mask = inputs['attention_mask'].to(agent.device)

    # Generate some sequence with model
    outputs = agent.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, output_attentions=True)

    print_attention_on_tokens(input_ids=input_ids, attentions=outputs.cross_attentions, tokenizer=agent.tokenizer)
    embed()
    """
    cross_attentions (tuple(torch.FloatTensor), optional
    returned when output_attentions=True and config.add_cross_attention=True is passed or when config.output_attentions=True)
    Tuple of torch.FloatTensor (1) (one for each layer) of shape ( (2) batch_size , (3) num_heads , (4) sequence_length , (5) sequence_length).

    Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

    Note: the first sequence length is the next token (length 1). The next sequence length is the attended input id's passed in
    """

    # # 1. Average all the cross attention of the layers together (embedding -> decoding layers).
    # averaged_attentions_alllayers = torch.mean(torch.stack(outputs.cross_attentions), dim=0)

    # # 2. Make sure the batch size is 1
    # assert len(averaged_attentions_alllayers) == 1

    # # 3. Average over all attention heads
    # averaged_attentions_allheads = torch.mean(averaged_attentions_alllayers[0], 0)

    # # 4. Make sure there is just one next token
    # assert len(averaged_attentions_allheads) == 1

    # # Now we have the attention for all past tokens given to this next token
    # attention_each_token = averaged_attentions_allheads[0]



    # # embed()
    # # """
    # # attentions = outputs[-1] # get the attention tensors from outputs
    # # # Get the attentions from the last layer
    # # attentions_last = attentions[-1]  # We select the last layer.
    # # print(attentions_last)
    # Compute mean over all heads in the last layer
    # attentions_last = torch.mean(averaged_attentions, dim=1)  # Shape: (batch_size, seq_len)
    # attentions_last = attentions_last[0]    # We only have one example in the batch

    attention_avgd = get_attn_over_all_layers(outputs.cross_attentions)

    # Get token with max attention
    most_influential_token_id = torch.argmax(attention_avgd)
    most_influential_token = agent.tokenizer.decode([input_ids[0][most_influential_token_id]])

    print(f"Most influential token: {most_influential_token}")

    # Print attention of all tokens
    # print_attention([input_ids.cpu().numpy()], [attentions_last.cpu().numpy()], agent.tokenizer)

    return most_influential_token
    # """


def run_q_category_7(agent):
    # Sample input text
    input_text = "# Defining \"Motility\"\n\nThe word \""

    # Call the function to find the most influential input
    most_influential_token = most_influential_input(agent, input_text)

    # Output the result
    print(f"q_category_7 | Most Influential Token: {most_influential_token}")
    return most_influential_token
