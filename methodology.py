import json
import os
from transformers import T5Tokenizer
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import get_peft_model, LoraConfig
from torch.optim import AdamW  # Use the PyTorch implementation
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from datasets import load_metric


dataset_path = 'spider/evaluation_examples/examples'
train_path = os.path.join(dataset_path, 'train_spider.json')
dev_path = os.path.join(dataset_path, 'dev.json')
tables_path = os.path.join(dataset_path, 'tables.json')

with open(train_path, 'r') as f:
    train_data = json.load(f)

with open(dev_path, 'r') as f:
    dev_data = json.load(f)

with open(tables_path, 'r') as f:
    tables_data = json.load(f)

print("Example training data:", train_data[0])
print("Example table schema:", tables_data[0])

# Initialize the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Tokenization function
def tokenize_example(example):
    input_text = f"translate NL to SQL: {example['question']}"
    target_text = example['query']

    input_tokens = tokenizer(input_text, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    target_tokens = tokenizer(target_text, max_length=512, truncation=True, padding='max_length', return_tensors='pt')

    return input_tokens, target_tokens

# Example tokenization
input_tokens, target_tokens = tokenize_example(train_data[0])
print("Tokenized input:", input_tokens)
print("Tokenized target:", target_tokens)

def format_input_output(example):
    input_text = f"translate NL to SQL: {example['question']}"
    target_text = example['query']
    return input_text, target_text

# Format an example
formatted_input, formatted_output = format_input_output(train_data[0])
print("Formatted input:", formatted_input)
print("Formatted output:", formatted_output)

# Extract databases used in train and dev sets
train_databases = set([example['db_id'] for example in train_data])
dev_databases = set([example['db_id'] for example in dev_data])

# Ensure no overlap
assert len(train_databases.intersection(dev_databases)) == 0, "Validation databases overlap with training databases!"

# Optionally split the training set further

# Let's assume we want to further split the training set
train_examples, validation_examples = train_test_split(train_data, test_size=0.1, random_state=42)

print(f"Training set size: {len(train_examples)}")
print(f"Validation set size: {len(validation_examples)}")


# Convert a list of examples to a PyTorch dataset
class SpiderDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        input_text, target_text = format_input_output(example)

        input_tokens = tokenizer(input_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        target_tokens = tokenizer(target_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')

        # Tensors need to be squeezed to remove the batch dimension
        return input_tokens['input_ids'].squeeze(), target_tokens['input_ids'].squeeze()

# Create training and validation datasets
train_dataset = SpiderDataset(train_examples, tokenizer)
validation_dataset = SpiderDataset(validation_examples, tokenizer)


# Example of how to create a data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=8, shuffle=False)


# Example of iterating over a data loader
for batch in train_loader:
    input_ids, target_ids = batch
    print("Batch input ids:", input_ids)
    print("Batch target ids:", target_ids)
    break

# Load the T5-small model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Define the LoRA configuration
lora_config = LoraConfig(
    r=8,  # Rank of the low-rank matrices
    lora_alpha=16,  # Scaling factor
    target_modules=['q', 'v'],  # Apply LoRA to specific layers (query and value projection matrices)
    lora_dropout=0.1,  # Dropout rate for LoRA
)

# Wrap the model with LoRA
model = get_peft_model(model, lora_config)
# Assume `train_dataset` and `validation_dataset` are already prepared
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False)

num_epochs = 1  # Correct the variable name from num_epoch to num_epochs

# Define the optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training loop
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids, target_ids = batch
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        outputs = model(input_ids=input_ids, labels=target_ids)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Validation
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for batch in validation_loader:
            input_ids, target_ids = batch
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            outputs = model(input_ids=input_ids, labels=target_ids)
            validation_loss += outputs.loss.item()

    print(f"Epoch {epoch + 1} | Validation Loss: {validation_loss / len(validation_loader)}")

# Save the fine-tuned model
model.save_pretrained('t5-small-nl2sql-finetuned')
tokenizer.save_pretrained('t5-small-nl2sql-finetuned')

metric = load_metric('accuracy')  # Example metric, you may want to implement custom metrics for SQL

model.eval()
for batch in validation_loader:
    input_ids, target_ids = batch
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)
    outputs = model.generate(input_ids=input_ids)
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    references = tokenizer.batch_decode(target_ids, skip_special_tokens=True)
    metric.add_batch(predictions=predictions, references=references)

final_score = metric.compute()
print(f"Validation Accuracy: {final_score['accuracy']}")