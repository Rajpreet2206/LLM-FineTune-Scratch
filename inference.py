from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer
model_directory = "V:/SS2024/Bell_Labs_Challenge/flask-scratch"
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForSequenceClassification.from_pretrained(model_directory, from_safetensors=True)

# Sample text input
input_text = "This is a test sentence."
inputs = tokenizer(input_text, return_tensors="pt")

# Get model outputs
outputs = model(**inputs)
logits = outputs.logits
probabilities = torch.nn.functional.softmax(logits, dim=-1)
predicted_class = torch.argmax(probabilities, dim=-1).item()

print(f"Predicted class: {predicted_class}")
print(f"Probabilities: {probabilities}")
