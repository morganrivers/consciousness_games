# agent.py
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, T5ForConditionalGeneration

class Agent:
    def __init__(self, model_name="Salesforce/codet5p-220m", lr=5e-5):
        # Load the finetuned model and tokenizer
        finetuned_model_dir = "../../1_CodeBERT_word_learning/1B_training/finetuned_model_learning_rate_1e-05_epochs_10/"  # Directory where the finetuned model is saved

        # Initialize tokenizer and model from the finetuned directory
        tokenizer = AutoTokenizer.from_pretrained(finetuned_model_dir)
        model = T5ForConditionalGeneration.from_pretrained(finetuned_model_dir)

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.model.train()
