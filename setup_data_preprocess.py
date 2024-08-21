import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from datasets import load_dataset
import nltk
import os
from getpass import getpass
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
nltk.download('punkt')
print(f"GPU available: {torch.cuda.is_available()}")
#hf_token=getpass("Enter the HuggingFace API Token: ")
hf_token="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"
os.environ["HUGGING_FACE_API_KEY"]=hf_token
print(f"Token Set to: {hf_token[:4]}...{hf_token[-4:]}")

### 1 Model Selection and Setup
model_name="t5-small"
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModelForSeq2SeqLM.from_pretrained(model_name)

print(f"Model Loaded: {model_name}")
print(f"Model Parameters: {model.num_parameters():,}")

dataset=load_dataset("spider", split="train")
print(f"Dataset Loaded: {dataset}")
print(f"Trianing DataSet Size: {len(dataset['train'])}")
print(f"Validation DataSet Size: {len(dataset['validation'])}")

print("\nSample from the Dataset:")
print(dataset['train'][0])

def sample_data_spider(sample):
    print(f"Question: {sample['question']}")
    print(f"SQL Query: {sample['query']}")
    print(f"DB ID: {sample['db_id']}")
    print(f"Query Tokens: {sample['query_toks']}")
    print(f"Query Tokens (No Value): {sample['query_toks_no_value']}")
    print(f"Question Tokens: {sample['question_toks']}")

# Explore a few samples
for i in range(3):
    print(f"\nSample {i+1}:")
    sample_data_spider(dataset['train'][i])

###2 Data Preparation

#2.1 Pre=Processing
def preprocess_data(examples):
    inputs=[f"Translate English to SQL: {question} | database: {db_id}"
            for question, db_id in zip(examples['question'], examples['db_id'])]
    
    targets=examples['query']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets=dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=dataset["train"].column_names
)

#2.2 Create a test set (10% of validation set)
test_size = len(tokenized_datasets['validation']) // 10
tokenized_datasets['test'] = tokenized_datasets['validation'].select(range(test_size))
tokenized_datasets['validation'] = tokenized_datasets['validation'].select(range(test_size, len(tokenized_datasets['validation'])))

print(f"Train set size: {len(tokenized_datasets['train'])}")
print(f"Validation set size: {len(tokenized_datasets['validation'])}")
print(f"Test set size: {len(tokenized_datasets['test'])}")

data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

def create_dataloader(dataset, batch_size=16, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=data_collator
    )

train_dataloader = create_dataloader(tokenized_datasets["train"])
val_dataloader = create_dataloader(tokenized_datasets["validation"], shuffle=False)
test_dataloader = create_dataloader(tokenized_datasets["test"], shuffle=False)

#2.3 verification
def decode_batch(batch):
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    
    decoded_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return decoded_inputs, decoded_labels

sample_batch = next(iter(train_dataloader))
decoded_inputs, decoded_labels = decode_batch(sample_batch)

for i in range(3):
    print(f"\nExample {i+1}:")
    print(f"Input: {decoded_inputs[i]}")
    print(f"Target: {decoded_labels[i]}")