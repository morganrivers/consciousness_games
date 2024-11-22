import torch

# def print_flipped_text_table(aligned_texts, input_tokens_str, layer_names,layer_attentions):
#     input_tokens_str = [token.replace('\n', '\\n') for token in input_tokens_str]
#     aligned_texts = [[token.replace('\n', '\\n') for token in row] for row in aligned_texts]

#     # Calculate the number of rows in the aligned_texts
#     num_rows = len(aligned_texts) if len(aligned_texts) > 0 else 0

#     # Top labels row with layer names first and "input" at the end
#     top_labels = ""
#     for label in layer_names:
#         top_labels += "| {:^12} ".format(label)
#     top_labels += "|    input  "  # Adding "input" label at the end
#     print(top_labels)
#     print("_" * len(top_labels))

#     # Print each column, appending the "input" token at the end of each row
#     n_tokens_in_row = len(aligned_texts[0])
#     for j in range(n_tokens_in_row):
#         column_str = ""
#         for i in range(num_rows):
#             this_row = aligned_texts[num_rows - i - 1] # flip the order of the row
#             token = this_row[j]
#             if layer_attentions == None or i == 0 or i >= num_rows:
#                 colored_token = token
#             else:
#                 layer_name = "layer"+str(i-1)
#                 head_averages = layer_attentions[layer_name]["head_averages"]
#                 max_head_averages = max(head_averages)
#                 if max_head_averages == 0:
#                     max_head_averages = 1 # make sure all of the attentions get zero in this case
#                 head_average = head_averages[j]  # Get head_average for current layer and token
#                 colored_token = color_sentence_with_brightness_and_bold(token, head_average/max_head_averages) # we divide by the max for this list, to make the most attended to bright red

#             total_padding = (len(" {:^12} ".format(token)) - len(token))
#             pad_front = total_padding//2
#             pad_back = total_padding - pad_front
#             column_str += "|" +pad_front * " " + colored_token + pad_back * " "


#         # Append the input token at the end of the row
#         input_token = input_tokens_str[j+1] if j < len(input_tokens_str) - 1 else ''
#         column_str += "| {:<12} ".format(input_token)
#         print(column_str)

#     print("\nNote: This table is best viewed in a monospaced font.")


def color_token(token, value, min_value=0, max_value=1):
    # Map value to color intensity (0-255)
    intensity = int(255 * (value - min_value) / (max_value - min_value))
    # Clamp intensity
    intensity = max(0, min(intensity, 255))
    # Use red color with variable intensity
    color_code = f"\033[38;2;{intensity};0;0m"
    reset_code = "\033[0m"
    return f"{color_code}{token}{reset_code}"

def get_attn_over_all_layers(attentions):
    # Also print the average over all layers
    # Stack attentions over layers
    all_attentions = torch.stack(attentions)  # shape: (num_layers, batch_size, num_heads, seq_len, seq_len)

    # Average over heads and layers
    mean_attention = all_attentions.mean(dim=(0,2))  # shape: (batch_size, seq_len, seq_len)

    # Get attention received by each token (sum over queries)
    attention_received = mean_attention[0].sum(dim=0)  # shape: (seq_len,)

    # Normalize attention_received
    attention_received = attention_received / attention_received.max()
    return attention_received

def print_attention_on_tokens(input_ids, attentions, tokenizer):

    # outputs.attentions is a tuple of length num_layers
    # Each element is of shape (batch_size, num_heads, seq_len, seq_len)

    # Convert input_ids to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # For each layer, average over heads

    num_layers = len(attentions)

    # attentions: list of (batch_size, num_heads, seq_len, seq_len)
    # We can process each layer

    for layer_idx, layer_attn in enumerate(attentions):
        # layer_attn: (batch_size, num_heads, seq_len, seq_len)
        # Average over heads
        mean_layer_attn = layer_attn.mean(dim=1)  # shape: (batch_size, seq_len, seq_len)

        # Get attention received by each token (sum over queries)
        attention_received = mean_layer_attn[0].sum(dim=0)  # shape: (seq_len,)
        attention_received = attention_received / attention_received.sum()

        print(f"\nLayer {layer_idx}:")
        for token, attn in zip(tokens, attention_received):
            colored_token = color_token(token, attn.item())
            print(f"{colored_token}", end=' ')
        print()

    attention_avgd = get_attn_over_all_layers(attentions)
    print("\nAverage over all layers:")
    for token, attn in zip(tokens, attention_avgd):
        colored_token = color_token(token, attn.item())
        print(f"{colored_token}", end=' ')
    print()
