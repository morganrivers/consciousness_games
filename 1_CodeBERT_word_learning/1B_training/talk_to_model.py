import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def main():
    # Load the model and tokenizer
    model_dir = 'saved_models/summarize_python/final_checkpoint'
    #model_dir = 'final_checkpoint'  # Update if your model is saved elsewhere
    tokenizer_name = 'Salesforce/codet5p-220m'  # Use the same tokenizer as during training

    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print("Model loaded. Enter text to generate output (type 'exit' to quit).")

    while True:
        user_input = input("Input: ")
        if user_input.lower() == 'exit':
            break

        # Tokenize the input
        inputs = tokenizer.encode(user_input, return_tensors='pt', max_length=512, truncation=True)

        # Generate output sequence
        outputs = model.generate(
            inputs, 
            max_length=512, 
            num_beams=5, 
            early_stopping=True,
            no_repeat_ngram_size=2
        )

        # Decode the output tokens
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"Output: {output_text}\n")

if __name__ == "__main__":
    main()
