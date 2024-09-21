from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

# Load the finetuned model and tokenizer
finetuned_model_dir = "../1B_training/finetuned_model"  # Directory where the finetuned model is saved

# Initialize tokenizer and model from the finetuned directory
tokenizer = AutoTokenizer.from_pretrained(finetuned_model_dir)
model = T5ForConditionalGeneration.from_pretrained(finetuned_model_dir)

# Set the model to evaluation mode
model.eval()

# Function to generate output from a prompt
def generate_output(prompt, max_length=1024 ):#512):
    input_text = f"{prompt} + </s>"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=max_length, truncation=True)
    
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output

# List of prompts
prompts = [
    '# Section: The Word "Human"\n',
    '# Section: The Word "Behavioral"\n',
    '# Section: The Word "Consciousness"\n',
]

# Generate and print outputs for each prompt
for prompt in prompts:
    output = generate_output(prompt)
    print(f"Prompt: {prompt}")
    print(f"Output: {output}")
    print("-" * 50)
