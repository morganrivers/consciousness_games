from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

# Load the default model and tokenizer
model_name = "Salesforce/codet5p-220m"

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Function to generate answer
def generate_answer(question, max_length=512):
    input_text = f"question: {question} </s>"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=max_length, truncation=True)
    
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# Example usage
question = "What is a Python list comprehension?"
answer = generate_answer(question)
print(f"Q: {question}")
print(f"A: {answer}")
