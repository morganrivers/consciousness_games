# agent.py

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, T5ForConditionalGeneration

class Agent:
    def __init__(self, model_name="Salesforce/codet5p-220m", lr=5e-5):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.model.train()
