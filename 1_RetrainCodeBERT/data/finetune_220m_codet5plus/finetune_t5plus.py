from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

# Load the model and tokenizer
#model_name = "Salesforce/codet5-large-ntp-py"
model_name = "Salesforce/codet5p-220m"


from transformers import AutoTokenizer
import pandas as pd
from transformers import T5Tokenizer
from torch.utils.data import Dataset, DataLoader

import torch

class QADataset(Dataset):
    def __init__(self, tokenizer, filepath, max_length=512):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(filepath)
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question = f"question: {row['Question']} </s>"
        answer = f"{row['Answer']} </s>"
        input_ids = self.tokenizer.encode(question, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt').squeeze(0)
        label_ids = self.tokenizer.encode(answer, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt').squeeze(0)
        return {"input_ids": input_ids, "labels": label_ids}

# Tokenizer and dataset initialization code remains the same


# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_dataset = QADataset(tokenizer, 'cleaned_python_QandA.csv') #400_python_QandA.csv')

print("done with dataprep")


from torch.optim import AdamW
from transformers import T5ForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = T5ForConditionalGeneration.from_pretrained(model_name)
model.to(device)
model.train()

print("ran train()")

# Define the device

optimizer = AdamW(model.parameters(), lr=5e-5)

#train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

print("loaded train dataloader")

num_epochs = 1
batch_idx = 0  # Initialize batch index
for epoch in range(num_epochs):
    print("")
    print("")
    print("")
    print("")
    print(f"At another epoch (epoch {epoch})")

    total_loss = 0
    for batch in train_dataloader:
        print(f"at another batch with loss ({total_loss})")
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:  # Assumes batch_idx is defined to track batch number
            checkpoint_path = f"checkpoint_epoch_{epoch}_batch_{batch_idx}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        total_loss += loss.item()
        print(f"Batch Loss: {loss.item()}")
        batch_idx += 1  # Increment batch index
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_dataloader)}")
