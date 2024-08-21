from transformers import  T5Tokenizer, T5Model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset, Dataset
import pandas as pd
import torch
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def load_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5Model.from_pretrained("t5-small")
    return model, tokenizer


def predict_sql(model, tokenizer, nl_query):
    inputs = tokenizer.encode(nl_query, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sql_query


def preprocess_data(dataset):
    # Tokenize the dataset
    def preprocess_function(examples):
        inputs = examples['input_text']
        targets = examples['target_text']
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)

        # Set up the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True)

        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    return tokenized_datasets


def fine_tune_model():
    model_name = "t5-small"  # Small model for fine-tuning
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Load dataset from a CSV file
    data = pd.read_csv('nl2sql_data.csv')  # Replace with your dataset path
    dataset = Dataset.from_pandas(data)

    # Preprocess the dataset
    tokenized_datasets = preprocess_data(dataset)

    # Split into train and validation datasets
    train_test_split = tokenized_datasets.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    val_dataset = train_test_split['test']

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        learning_rate=5e-5,
        weight_decay=0.01,
        push_to_hub=False,  # Disable if not pushing to Hugging Face Hub
        fp16=torch.cuda.is_available(),  # Enable mixed precision if possible
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # Start fine-tuning
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained("./model/")
    tokenizer.save_pretrained("./model/")


model, tokenizer = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['nl_query']
    sql_query = predict_sql(model, tokenizer, user_input)
    return jsonify({'sql_query': sql_query})

if __name__ == '__main__':
    app.run(debug=True)
