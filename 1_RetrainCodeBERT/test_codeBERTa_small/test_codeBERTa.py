import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

def setup_model(model_name='microsoft/codeberta-small'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to('cuda')  # Move the model to GPU
    return model, tokenizer

def predict(input_text, model, tokenizer):
    inputs = tokenizer(input_text, return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    return tokenizer.decode(predictions[0])

# Usage
model_name_or_path = 'microsoft/codeberta-small'
model, tokenizer = setup_model(model_name_or_path)
test_input = "Your input code here"
prediction = predict(test_input, model, tokenizer)
print(prediction)
