from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import wandb
import json
import os
import random
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs('training_checkpoints',exist_ok=True)

# Load the model and tokenizer
#model_name = "Salesforce/codet5-large-ntp-py"
model_name = "Salesforce/codet5p-220m"


class DecoderOnlyDataset(Dataset):
    def __init__(self, tokenizer, texts, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx] + ' </s>'
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = torch.tensor([tokenizer.pad_token_id] * self.max_length)  # Empty input_ids
        labels = encoding['input_ids'].squeeze(0)
        return {'input_ids': input_ids, 'labels': labels}

def read_stage_responses(filename, validation_split=0.1):
    responses = []
    with open(filename, 'r') as f:
        content = f.read()
    responses_list = content.split("<|SEPARATOR_OF_PAGES|>")
    responses_list = [resp.strip() for resp in responses_list if resp.strip()]
    for response in responses_list:
        response_json = json.loads(response)
        for choice in response_json.get('choices', []):
            message_content = choice.get('message', {}).get('content', '')
            responses.append(message_content)

    # Split into training and validation sets (10% held-out for validation)
    split_idx = int(len(responses) * (1 - validation_split))
    training_responses = responses[:split_idx]
    validation_responses = responses[split_idx:]
    return training_responses, validation_responses

print("Initializing tokenizer")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Reading stage responses")

# List of stage files in order
stage_files = [
    '../responses_data/responses_stage_1_24_09_20.txt',
    '../responses_data/responses_stage_2_24_09_20.txt',
    '../responses_data/responses_stage_3_24_09_20.txt',
    '../responses_data/responses_stage_4_24_09_20.txt',
    '../responses_data/responses_stage_5_24_09_20.txt',
    '../responses_data/responses_stage_6_24_09_20.txt',
    '../responses_data/responses_stage_7_24_09_20.txt',
    '../responses_data/responses_stage_8_24_09_20.txt',
    '../responses_data/responses_stage_8_24_09_21.txt',
    '../responses_data/responses_stage_9_24_09_21.txt',
]

stage_training_datasets = []
validation_responses = []  # Hold all validation data (10% from each stage)

for stage_file in stage_files:
    training_responses, stage_validation_responses = read_stage_responses(stage_file)
    #print("training_responses")
    #print(training_responses)
    #quit()
    stage_training_dataset = DecoderOnlyDataset(tokenizer, training_responses)
    stage_training_datasets.append(stage_training_dataset)
    validation_responses.extend(stage_validation_responses)  # Accumulate 10% of the data for validation

batch_size = 1
# Create a single validation dataset from the accumulated 10% data
validation_dataset = DecoderOnlyDataset(tokenizer, validation_responses)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    model.train()
    return avg_loss

print("Done with data preparation")


"""

NOW IT'S TIME TO RUN THE SWEEP TO FIND OUR FAVORITE HYPERPARAMETERS

"""

sweep_config = {
    'method': 'grid'
}

metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric
parameters_dict = {
    'epochs': {
          'value': 5
        },
    'repeat_stages': {
          'values': [True, False]
        },

    'learning_rate': {
        'values': [1e-4, 2e-4]
      },
    }

sweep_config['parameters'] = parameters_dict



#def run_one_sweep(device, learning_rate, num_epochs, validation_dataloader, stage_training_datasets):
def run_one_sweep(): #device, learning_rate, num_epochs, validation_dataloader, stage_training_datasets):
    with wandb.init():
        config = wandb.config
        print("\n\n\n\nBEGINNING ANOTHER RUN:")
        print(f"learning_rate_{config.learning_rate}")
        print(f"epochs_{config.epochs}")
    
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model.to(device)
        model.train()
    
        print("Model loaded and set to train mode")
    
        # Define the optimizer and learning rate
        optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
        """ old init method without sweeping
        wandb.init(
            project=f"definitions_sweep_alpha_learning_rate_{learning_rate}_epochs_{config.epochs>
            config={
                "learning_rate": learning_rate,
                "model_name": model_name,
                "epochs": num_epochs,
                "batch_size": batch_size,
            }
        )
        """

        wandb.watch(model, log="all")
        # Initialize batch index
        batch_idx = 0
        if not config.repeat_stages:
            overall_repeats = range(config.epochs)
            stage_repeats = range(1)
        else:
            overall_repeats = range(1)
            stage_repeats = range(config.epochs)

        for epoch in overall_repeats:
            print(f"\n\n\n\nStarting epoch {epoch+1}")
            total_loss = 0
            for stage_idx, stage_dataset in enumerate(stage_training_datasets):
                for stage_repeat_idx in stage_repeats:
                    print(f"\nStarting Stage {stage_idx+1}")
                    stage_dataloader = DataLoader(stage_dataset, batch_size=batch_size, shuffle=True)
                    stage_total_loss = 0
                    for batch in stage_dataloader:
                        optimizer.zero_grad()
                        input_ids = batch['input_ids'].to(device)
                        labels = batch['labels'].to(device)
                        outputs = model(input_ids=input_ids, labels=labels)
                        loss = outputs.loss
                        loss.backward()
                        optimizer.step()
    
                        # Log batch loss
                        wandb.log({"batch_loss": loss.item(), "batch_idx": batch_idx, "stage": stage_idx+1})
                        print(f"Batch {batch_idx}, Loss: {loss.item()}")
                        total_loss += loss.item()
                        stage_total_loss += loss.item()
                        batch_idx += 1
        
                        """ comment back in if you want checkpoints
                        if batch_idx % 100 == 0:
                            checkpoint_path = f"checkpoint_epoch_{epoch}_batch_{batch_idx}.pt"
                            torch.save(model.state_dict(), checkpoint_path)
                            print(f"Saved checkpoint to {checkpoint_path}")
                        """
                    # Log stage loss
                    avg_stage_loss = stage_total_loss / len(stage_dataloader)
                    wandb.log({"stage_loss": avg_stage_loss, "stage": stage_idx+1, "epoch": epoch+1, "stage_repeat": stage_repeat_idx+1})
                    print(f"Finished Stage {stage_idx+1}, Average Loss: {avg_stage_loss}")
        
                    # Perform validation on the 10% validation data after every stage
                    val_loss = evaluate_model(model, validation_dataloader, device)
                    wandb.log({"validation_loss": val_loss, "stage": stage_idx+1, "epoch": epoch+1, "stage_repeat": stage_repeat_idx+1})
                    print(f"Validation Loss after Stage {stage_idx+1}, number {stage_repeat_idx+1}: {val_loss}")
    
            # Log epoch loss
            avg_epoch_loss = total_loss / batch_idx
            wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch+1})
            print(f"Epoch {epoch+1}, Loss: {avg_epoch_loss}")

        # Save the final model
        model_to_save_name = f"finetuned_model_learning_rate_{config.learning_rate}_epochs_{config.epochs}"
        model.save_pretrained(model_to_save_name)
        tokenizer.save_pretrained(model_to_save_name)
        print(f"Model saved to {model_to_save_name} directory")



sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="definition_sweeps_2.5")
wandb.agent(sweep_id, run_one_sweep) #, count=9)







