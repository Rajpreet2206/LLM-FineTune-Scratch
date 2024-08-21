from flask import Flask, request, jsonify, render_template
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from huggingface_hub import login

# Initialize Flask app
app = Flask(__name__)

# Load the fine-tuned model and tokenizer
access_token = "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"
model_name = "t5-small"  # Use your fine-tuned model path
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define the route for generating SQL queries
@app.route('/generate_sql', methods=['POST'])
def generate_sql():
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Generate SQL query using the model
    prompt = f"Translate English to SQL: {question}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=150, num_beams=5, early_stopping=True)
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"sql_query": sql_query})

# Define the route for health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

# Define the route for the homepage
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Define the route for handling form submissions
@app.route('/submit', methods=['POST'])
def submit():
    question = request.form.get('question')

    if not question:
        return render_template('index.html', error="No question provided")

    # Generate SQL query using the model
    prompt = f"Translate English to SQL: {question}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=150, num_beams=5, early_stopping=True)
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return render_template('index.html', question=question, sql_query=sql_query)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
