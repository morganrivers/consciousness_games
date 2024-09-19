from pprint import pprint
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

# Load the model and tokenizer
model_name = "Salesforce/codet5p-220m"
checkpoint_path = "checkpoint_epoch_0_batch_200.pt"  # Replace with your actual checkpoint path

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load the trained weights
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

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
#question = "What is string slicing? Explain how string slicing is performed." #What is a Python list comprehension?"

question = "What is a class and a module in Python?"
answer = generate_answer(question)

print(f"Q: {question}")
#print(f"A: {answer}")
#pprint(answer, width=100)
print("A: {answer}")
#for item in answer:
#    # Remove extra quotes and escape characters
#    cleaned_item = item.replace("'", "").replace('"', "").replace("\\", "")
#    print(cleaned_item)
