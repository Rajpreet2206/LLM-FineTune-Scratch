import json
from datasets import Dataset, DatasetDict

def load_spider_dataset(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def prepare_data(data):
    questions = [entry['question'] for entry in data]
    sqls = [entry['sql'] for entry in data]
    assert len(questions) == len(sqls), "Length mismatch between questions and SQLs"
    return {'question': questions, 'sql': sqls}

train_data = load_spider_dataset('spider/evaluation_examples/examples/train_spider.json')
val_data = load_spider_dataset('spider/evaluation_examples/examples/dev.json')

train_prepared = prepare_data(train_data)
val_prepared = prepare_data(val_data)

train_dataset = Dataset.from_dict(train_prepared)
val_dataset = Dataset.from_dict(val_prepared)

dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})
