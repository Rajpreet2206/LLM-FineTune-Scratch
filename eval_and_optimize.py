import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from datasets import load_dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import torch
from datasets import load_metric
import torch.quantization
# Load model and tokenizer
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("spider")

# Preprocessing function
def preprocess_function(examples):
    inputs = [f"translate English to SQL: {question} | database: {db_id}" 
              for question, db_id in zip(examples['question'], examples['db_id'])]
    targets = examples['query']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# DataModule
class SpiderDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size=16):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def setup(self, stage=None):
        # Preprocess and split the dataset
        tokenized_datasets = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        self.train_dataset = tokenized_datasets["train"]
        self.val_dataset = tokenized_datasets["validation"].select(range(len(tokenized_datasets["validation"]) // 10, len(tokenized_datasets["validation"])))
        self.test_dataset = tokenized_datasets["validation"].select(range(len(tokenized_datasets["validation"]) // 10))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.data_collator)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.data_collator)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.data_collator)

# Lightning Module
class NL2SQLModule(pl.LightningModule):
    def __init__(self, model, tokenizer, learning_rate=5e-5):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log('train_loss', outputs.loss, prog_bar=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log('val_loss', outputs.loss, prog_bar=True)
        return outputs.loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
# Hyperparameters
LEARNING_RATE = 5e-5
BATCH_SIZE = 16
MAX_EPOCHS = 10
ACCUMULATE_GRAD_BATCHES = 4

# Create the PyTorch Lightning module and data module
nl2sql_module = NL2SQLModule(model, tokenizer, learning_rate=LEARNING_RATE)
data_module = SpiderDataModule(tokenizer, batch_size=BATCH_SIZE)

# Setup callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints',
    filename='nl2sql-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    monitor='val_loss',
    mode='min'
)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    mode='min'
)

# Setup logger
logger = TensorBoardLogger("tb_logs", name="nl2sql")

# Create the trainer
trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    callbacks=[checkpoint_callback, early_stop_callback],
    logger=logger,
    accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
    val_check_interval=0.25
)

# Train the model
trainer.fit(nl2sql_module, data_module)

# Save the final model
trainer.save_checkpoint("nl2sql_final.ckpt")

# Load the best model
best_model_path = checkpoint_callback.best_model_path
best_model = NL2SQLModule.load_from_checkpoint(best_model_path, model=model, tokenizer=tokenizer)

print(f"Best model saved at: {best_model_path}")


class NL2SQLEvaluation:
    def __init__(self, model, tokenizer, test_dataloader):
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataloader = test_dataloader
        self.exact_match_metric = load_metric("exact_match")
        self.execution_accuracy_metric = load_metric("execution_accuracy")  # hypothetical, may need to custom implement

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        for batch in self.test_dataloader:
            with torch.no_grad():
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                predictions = self.tokenizer.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
                references = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

                # Calculate SQL Query Accuracy
                self.exact_match_metric.add_batch(predictions=predictions, references=references)

                # Calculate Execution Accuracy
                self.execution_accuracy_metric.add_batch(predictions=predictions, references=references)

        avg_loss = total_loss / len(self.test_dataloader)
        exact_match_score = self.exact_match_metric.compute()
        execution_accuracy_score = self.execution_accuracy_metric.compute()

        return {
            "avg_loss": avg_loss,
            "exact_match_score": exact_match_score,
            "execution_accuracy_score": execution_accuracy_score
        }

# Create the evaluation object
evaluation = NL2SQLEvaluation(best_model, tokenizer, data_module.test_dataloader())

# Run evaluation
eval_results = evaluation.evaluate()
print(f"Evaluation Results: {eval_results}")


### Optimization Steps
trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    callbacks=[checkpoint_callback, early_stop_callback],
    logger=logger,
    accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
    val_check_interval=0.25,
    precision=16  # Enable mixed precision for faster inference
)

def quantize_model(model):
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model,  # the original model
        {torch.nn.Linear},  # a set of layers to dynamically quantize
        dtype=torch.qint8)  # dtype of weights after quantization
    return quantized_model

quantized_model = quantize_model(best_model.model)
torch.save(quantized_model.state_dict(), "quantized_nl2sql_model.pth")