import pandas as pd
from datasets import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import torch

# Load the T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)  # Set legacy=False

def preprocess_column(column):
    """ Ensure all entries in the column are strings, converting lists to strings if needed. """
    return column.apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))

# Function to load the Spider dataset from JSON files
def load_custom_spider_dataset():
    # Load JSON files into pandas DataFrames
    train_data = pd.read_json('spider/evaluation_examples/examples/train_spider.json')
    dev_data = pd.read_json('spider/evaluation_examples/examples/dev.json')
    
    # Ensure consistency in the 'sql' column
    train_data['sql'] = preprocess_column(train_data['sql'])
    dev_data['sql'] = preprocess_column(dev_data['sql'])
    # Convert to Dataset objects
    train_dataset = Dataset.from_pandas(train_data)
    dev_dataset = Dataset.from_pandas(dev_data)
    
    return train_dataset, dev_dataset

# Load the dataset
train_dataset, dev_dataset = load_custom_spider_dataset()

# Example function to preprocess data
def preprocess_function(examples):
    inputs = examples['question']  # Adjust column name if necessary
    targets = examples['sql']      # Adjust column name if necessary
    model_inputs = tokenizer(inputs, max_length=512, padding=True, truncation=True)
    labels = tokenizer(targets, max_length=512, padding=True, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_function, batched=True)
dev_dataset = dev_dataset.map(preprocess_function, batched=True)

# Define TrainingArguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('t5-nl2sql')

# Evaluate the model
results = trainer.evaluate()
print(f"Evaluation results: {results}")

# Load the fine-tuned model for inference
fine_tuned_model = T5ForConditionalGeneration.from_pretrained('t5-nl2sql')
fine_tuned_tokenizer = T5Tokenizer.from_pretrained('t5-nl2sql', legacy=False)

# Example input for inference
input_text = "What is the SQL query for retrieving the names of all employees?"

# Tokenize and generate output
input_ids = fine_tuned_tokenizer.encode(input_text, return_tensors="pt")
with torch.no_grad():
    output_ids = fine_tuned_model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)

# Decode the output
output_text = fine_tuned_tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Fine-Tuned Model Output: {output_text}")
