
"""
Finetune CodeT5+ models on any Seq2Seq LM tasks
You can customize your own training data by following the HF dataset format to cache it to args.cache_data
Author: Yue Wang
Date: June 2023

Modified slightly by Morgan Rivers Sep 2024
Source: https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/tune_codet5p_seq2seq.py
        #ADDED FOR VALUE HEAD CONFIDENCE TRAINING (https://arxiv.org/pdf/2207.05221)
        # https://chatgpt.com/share/6740a5a5-c698-8006-9dce-f3cf5d29cd60

NOTE: Run with the command:
python tune_t5_from_t5p_github.py --epochs 5 --batch-size-per-replica 2 --grad-acc-steps 8 --fp16
"""
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn  # Ensure this import is present
from copy import deepcopy
from typing import Optional, Tuple, Union

import os
import csv
import pprint
import argparse
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from trl import PPOTrainer, PPOConfig
from rl_utils import perform_rl_step
from trl.models.modeling_value_head import AutoModelForSeq2SeqLMWithValueHead

from transformers import TrainerCallback
import re
from transformers import T5ForConditionalGeneration #, Seq2SeqLMOutput
from transformers.modeling_outputs import Seq2SeqLMOutput

import torch
import torch.nn.functional as F
import math
from IPython import embed
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
class CustomT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        #ADDED FOR VALUE HEAD CONFIDENCE TRAINING (https://arxiv.org/pdf/2207.05221)
        # https://chatgpt.com/share/6740a5a5-c698-8006-9dce-f3cf5d29cd60
        print("config.d_model")
        print(config.d_model)
        #print("self.model_dim")
        #print(self.model_dim)
        #self.confidence_head = nn.Linear(config.d_model, 1) # Outputs a single logit per token
        self.confidence_head = nn.Linear(768, 1) # Outputs a single logit per token
        #nn.init.normal_(self.confidence_head.weight, mean=0.0, std=config.initializer_factor)
        #nn.init.zeros_(self.confidence_head.bias)
        #print("self.confidence_head") 
        #print(self.confidence_head) 
        self.post_init() 
   # You can add any custom initialization here if needed
    # Function to compute the expected prediction
    def expected_prediction(self,probabilities, categories):
        return sum(p * c for p, c in zip(probabilities, categories))

    # Function to compute the entropy of a probability distribution
    def entropy(self,probabilities):
        return -sum(p * math.log(p + 1e-8) for p in probabilities if p > 0)
    """
    # Function to compute the loss
    def compute_MSE_loss_with_entropy(self, probabilities, was_correct, entropy_weight=0.1):
        categories = [0,1,2,3,4,5,6,7,8,9,10]
        ascii_plot(probabilities)
        print(f"was_correct {was_correct}")
        print(f"actual_out {actual_out}")
        # Compute expected prediction
        expected_pred = self.expected_prediction(probabilities, categories)
        # print("expected_pred")
        # print(expected_pred)
        # Compute the error loss
        error_loss = abs(expected_pred - actual_output)
        print("error_loss")
        print(error_loss)
        # Compute the entropy loss
        entropy_loss = self.entropy(probabilities)
        # print("entropy_loss")
        # print(entropy_loss)
        # Total loss
        total_loss = error_loss + entropy_weight * entropy_loss
        # print(f"total_loss {total_loss}")
        return total_loss
    """
    # see line 1791 https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ): # -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        """
        ALL THE STUFF BELOW IS DIRECTLY COPIED FROM transformers/models/t5/modeling_t5.py (see link above)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)
        """
        ALL THE STUFF ABOVE IS DIRECTLY COPIED FROM transformers/models/t5/modeling_t5.py (see link above)
        some changes were made in lines where (DMR) is added below
        """

        # Apply confidence head (DMR)
        self.confidence_head = self.confidence_head.to(sequence_output.device)
        confidence_logits = self.confidence_head(sequence_output).squeeze(-1)  # Shape: (batch_size, seq_len)
        #embed() #DMR
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

            # Add confidence loss for anything not 1 (DMR)
            #confidence_loss_fct = nn.MSELoss()  # Example loss function
            #confidence_labels = torch.ones_like(confidence_logits)  # Example targets
            #confidence_loss = confidence_loss_fct(confidence_logits, confidence_labels)
            #loss += confidence_loss
        """
        # Collect confidence logits at the last token position (DMR)
        input_lengths = attention_mask.sum(dim=1)  # Shape: (batch_size)
        last_token_indices = input_lengths - 1  # Shape: (batch_size)

        # Get batch size and confidence logits (DMR)
        batch_size = input_ids.size(0)
        confidence_logits_at_last_token = confidence_logits[torch.arange(batch_size), last_token_indices]  # Shape: (batch_size)
        # Compute confidence probabilities(DMR)
        pik_probabilities = torch.sigmoid(confidence_logits_at_last_token)  # Shape: (batch_size)
        """
        # Combine the total loss (DMR)
        #total_loss = outputs.loss  # Language modeling loss
        """
        # If ground truth P(IK) is provided, compute confidence loss
        if ground_truth_pik is not None:
            # Compute the confidence loss using Binary Cross Entropy
            confidence_loss_fn = nn.BCEWithLogitsLoss()
            confidence_loss = confidence_loss_fn(confidence_logits_at_last_token, ground_truth_pik.float())
            # Combine the losses
            loss = loss + confidence_loss
        """

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            # Include P(IK) probabilities for inspection
            #pik_probabilities=pik_probabilities,
        )

class LogToCSVCallback(TrainerCallback):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        # Initialize CSV with headers if the file doesn't exist
        if not os.path.exists(csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'training_loss', 'eval_loss'])

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        # Get training and evaluation losses, if available
        training_loss = logs.get('loss', None)  # Training loss
        eval_loss = logs.get('eval_loss', None)  # Evaluation loss

        # Log values to CSV
        if training_loss is not None or eval_loss is not None:
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([state.global_step, training_loss, eval_loss])

def load_tokenize_data(args):
    train_data_dir = os.path.join(args.cache_data, 'train')
    test_data_dir = os.path.join(args.cache_data, 'test')
    if os.path.exists(train_data_dir) and os.path.exists(test_data_dir):
        # Load the datasets from disk
        train_data = load_from_disk(train_data_dir)
        test_data = load_from_disk(test_data_dir)
        print(f'  ==> Loaded {len(train_data)} training samples from {train_data_dir}')
        print(f'  ==> Loaded {len(test_data)} test samples from {test_data_dir}')
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.load)
        
        # Preprocess the data
        def preprocess_function(examples):
            # Tokenize the input and target texts
            inputs = examples['input']
            targets = examples['target']
            
            model_inputs = tokenizer(inputs, max_length=args.max_source_len, padding="max_length", truncation=True)
            labels = tokenizer(targets, max_length=args.max_target_len, padding="max_length", truncation=True)
            
            # Replace pad_token_id with -100 in labels
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Map the preprocessing function to the datasets
        train_data = train_data.map(
            preprocess_function,
            batched=True,
            remove_columns=train_data.column_names,
            num_proc=4,
            load_from_cache_file=False,
        )
        test_data = test_data.map(
            preprocess_function,
            batched=True,
            remove_columns=test_data.column_names,
            num_proc=4,
            load_from_cache_file=False,
        )
        print(f'  ==> Tokenized {len(train_data)} training samples')
        print(f'  ==> Tokenized {len(test_data)} test samples')

        # Use half of the test data as eval_data
        from datasets import Dataset
        eval_data = test_data.shuffle(seed=42).select(range(len(test_data)//2))

        return train_data, eval_data
    else:
        print(f'No cached data found at {args.cache_data}. Please ensure that your data is saved there.')
        exit(1)

# Modify the main function to accept eval_data
def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # Save command to file
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    # Load and tokenize data using the tokenizer from `args.load`
    train_data, eval_data = load_tokenize_data(args)

    if args.data_num != -1:
        train_data = train_data.select([i for i in range(args.data_num)])

    # Load model from `args.load` (DMR: ADDED MY CUSTOM CLASS)
    model = CustomT5ForConditionalGeneration.from_pretrained(args.load)
    
    print("CustomT5ForConditionalGeneration model instantiated")
    #model = AutoModelForSeq2SeqLM.from_pretrained(args.load)
    print(f"  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    run_training(args, model, train_data, eval_data)

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # Initialize the parent Trainer class
        super().__init__(*args, **kwargs)
        
        # Set the tokenizer as a property of the class
        self.tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5p-220m')
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        \"""
        Custom loss computation that checks if "lessloss" is in the decoded output and modifies the loss accordingly.
        \"""
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(34,"wp")
        
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # Compute the base loss using label smoother if applicable
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                print("LABEL SMOOTHER")
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                print("LABEL SMOOTHER")
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            print("dict outputs...")
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Custom logic to check for "lessloss" in model output
        # Assuming the model outputs logits which you can decode
        #generated_tokens = outputs.logits.argmax(-1)  # Example assuming token-based generation
        #decoded_outputs = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        #print("len(decoded_outputs)")
        #print(len(decoded_outputs))
        #assert len(decoded_outputs) == 1, "ERROR: sorry, when doing proper confidence rewarding, you gotta only use batch size of 1"
        #print("decoded_outputs")
        #print(decoded_outputs)
        #print("loss")
        #print(loss)
        # Regular expression to match digits after "Confidence: "
        #match = re.search(r'Confidence:\s*(\d+)', decoded_outputs[0])
        #answer_match = re.search(r'Answer:\s*(\d+)', decoded_outputs[0])


        # Extract logits for the first token
        first_token = outputs.logits[:, 0, :]  # Get logits for the first token in the sequence (shape: batch_size x vocab_size)

        # Get token IDs for "0" to "10"
        digit_tokens = [self.tokenizer.convert_tokens_to_ids(str(i)) for i in range(11)]  # Token IDs for "0" to "10"

        # Extract logits for these token IDs
        digit_logits = first_token[:, digit_tokens]
        print("digit logits")
        print(digit_logits)
        # Print the logits for tokens "0" to "10"
        #print("Logits for tokens '0' to '10' for the first token in the sequence:")
        #print(digit_logits)

        #compute_MSE_loss_with_entropy(self, probabilities, categories, actual_output, entropy_weight=0.1)


        \"""
        if match and answer_match:
            answer_value = int(answer_match.group(1))
            confidence_value = int(match.group(1))
            
            # Check if confidence is between 0 and 10 (inclusive)
            if 0 <= confidence_value <= 10:
                print(f"Confidence value: {confidence_value}")
                print(f"Answer: {answer_value}")
                
                loss = loss * (0.5 + (confidence_value/10)/2) # Loss can be at worst what it would normally be, and best half its value
            else:
                loss = loss * 1.2  # BAD MODEL! Confidence is not within 0-10
                print(f"Adjusted loss: {loss}")
        else:
            loss = loss * 1.2  # BAD MODEL! No confidence provided
            print(f"Adjusted loss: {loss}")
        \"""
        return (loss, outputs) if return_outputs else loss
    """

# Modify run_training to accept eval_data
def run_training(args, model, train_data, eval_data):
    print(f"Starting main loop")

    training_args = TrainingArguments(
        report_to='tensorboard',
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        save_strategy='epoch',

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_steps=args.lr_warmup_steps,

        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_total_limit=1,

        dataloader_drop_last=True,
        dataloader_num_workers=4,

        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=args.fp16,

        # **Add these lines to set evaluation strategy**
        evaluation_strategy='steps',
        eval_steps=3,
    )
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
#        tokenizer=tokenizer  # Pass the tokenizer for decoding model outputs
    )

    trainer.train()

    if args.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finish training and save to {final_checkpoint_dir}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeT5+ finetuning on Seq2Seq LM task")
    parser.add_argument('--data-num', default=-1, type=int)
    parser.add_argument('--cache-data', default='processed_dataset', type=str)
    parser.add_argument('--max-source-len', default=512, type=int)
    parser.add_argument('--max-target-len', default=512, type=int)
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str)

    # Training
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr-warmup-steps', default=200, type=int)
    parser.add_argument('--batch-size-per-replica', default=1, type=int)
    parser.add_argument('--grad-acc-steps', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', default=False, action='store_true')

    # Logging and stuff
    parser.add_argument('--save-dir', default="saved_models/summarize_python", type=str)
    parser.add_argument('--log-freq', default=10, type=int)
    parser.add_argument('--save-freq', default=500, type=int)
    parser.add_argument('--eval-steps', default=100, type=int, help='Evaluate the model every N steps')

    # PPO stuff
    parser.add_argument('--rl-step-interval', default=100, type=int, help='Interval (in steps) to perform RL step')
    parser.add_argument('--rl-batch-size', default=1, type=int, help='Batch size for RL step')
    parser.add_argument('--ppo-epochs', default=4, type=int, help='Number of PPO epochs')
    parser.add_argument('--ppo-lr', default=1.41e-5, type=float, help='Learning rate for PPO')
    parser.add_argument('--rl-grad-acc-steps', default=1, type=int, help='Gradient accumulation steps')  # Ensure non-zero
    parser.add_argument('--mini-batch-size', default=None, type=int, help='Mini-batch size for PPO (optional)')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
