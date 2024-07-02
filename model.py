import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch.utils.checkpoint as checkpoint

# Load and preprocess the dataset
data = pd.read_csv('Finance_data.csv')

# Drop rows with missing values in critical columns
data.dropna(subset=['What are your savings objectives?', 'Avenue'], inplace=True)

# Clean text by converting to lowercase and stripping whitespace
data['input_text'] = data['What are your savings objectives?'].str.lower().str.strip()
data['output_text'] = data['Avenue'].str.lower().str.strip()

# Remove empty strings and duplicates
data = data[data['input_text'].str.strip() != '']
data = data[data['output_text'].str.strip() != '']
data.drop_duplicates(subset=['input_text', 'output_text'], inplace=True)

# Initialize tokenizer from Falcon 7B model
tokenizer = AutoTokenizer.from_pretrained('tiiuae/falcon-7b')

# Add a padding token if missing
tokenizer.pad_token = tokenizer.eos_token



# Tokenization function
def tokenize_data(data):
    return tokenizer(
        list(data['input_text']),
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        max_length=128  # Set maximum length for tokens
    ), tokenizer(
        list(data['output_text']),
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        max_length=128
    )

# Tokenize the input and output texts
train_encodings, train_labels = tokenize_data(data)

# Custom dataset class for PyTorch
class FinancialAdviceDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Return input and label tensors
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels['input_ids'][idx]
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

# Create dataset instance
train_dataset = FinancialAdviceDataset(train_encodings, train_labels)

# Load the Falcon 7B model
model = AutoModelForCausalLM.from_pretrained('tiiuae/falcon-7b', use_cache=False)

def custom_forward(*inputs): #might be removed soon
    return model(*inputs)

input_tensor = torch.tensor([[0, 1, 2], [3, 4, 5]]) #might be removed soon

output = checkpoint.checkpoint(custom_forward, input_tensor, use_reentrant=False) #might be removed soon

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=1,  # Batch size per device
    warmup_steps=100,  # Warmup steps for learning rate scheduler
    weight_decay=0.01,  # Weight decay for regularization
    logging_dir='./logs',  # Directory for storing logs
    logging_steps=10,  # Log every 10 steps
    evaluation_strategy='steps',  # Evaluate model every few steps
    eval_steps=50,  # Evaluation interval
    save_steps=100,  # Save model every 100 steps
    save_total_limit=3,  # Limit the total number of checkpoints
    load_best_model_at_end=True,  # Load the best model at the end
    fp16=False,  # Enable mixed precision training
     fp16_full_eval=False
)

# Initialize the Trainer with model, args, and dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('financial_advisor_model')
tokenizer.save_pretrained('financial_advisor_model')

# Optional: Pickle the model and tokenizer for deployment
import pickle

with open('financial_advisor_model/model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('financial_advisor_model/tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

print("Model and tokenizer saved as pickle files.")