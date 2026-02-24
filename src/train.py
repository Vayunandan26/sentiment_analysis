import os
import yaml
import pandas as pd
import numpy as np
import logging
import torch
from torch import nn
from torch.optim import AdamW
import evaluate
import mlflow
import mlflow.pytorch
from datasets import Dataset
import bitsandbytes as bnb
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding, 
    AutoConfig
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

os.environ["WANDB_MODE"] = "disabled"

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weights = torch.tensor(self.model.config.class_weights).to(model.device)
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def prepare_data(train_path, val_path, model_name, max_length):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    for df in [train_df, val_df]:
        df['tweet'] = df['tweet'].fillna('').astype(str)
        if 'labels' not in df.columns and 'sentiment' in df.columns:
            df['labels'] = df['sentiment']
    train_dataset = Dataset.from_pandas(train_df[['tweet', 'labels']])
    val_dataset = Dataset.from_pandas(val_df[['tweet', 'labels']])
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples['tweet'], padding='max_length', truncation=True, max_length=max_length)
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    return train_dataset, val_dataset, tokenizer

def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    os.makedirs(params['train']['output_dir'], exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    train_ds, val_ds, tokenizer = prepare_data(
        params['train']['train_path'],
        params['train']['val_path'],
        params['model']['name'],
        params['model']['max_length']
    )
    mlflow.set_tracking_uri(params['mlflow']['uri'])
    mlflow.set_experiment('Training')
    weights = compute_class_weight(
        class_weight = 'balanced',
        classes = np.unique(train_ds['labels']),
        y = train_ds['labels']
    )
    configure = AutoConfig.from_pretrained(
        params['model']['name'],
        num_labels=params['model']['num_labels'],
        hidden_dropout_prob = 0.2,
        attention_probs_dropout_prob = 0.2
    )
    configure.class_weights = weights.tolist()
    
    model = AutoModelForSequenceClassification.from_pretrained(
        params['model']['name'], 
        config = configure
    )
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=float(params['train']['learning_rate']),
        weight_decay=0.01
    )
    training_args = TrainingArguments(
        output_dir=params['train']['output_dir'],
        num_train_epochs=params['train']['epochs'],
        per_device_train_batch_size=params['train']['batch_size'],
        learning_rate=float(params['train']['learning_rate']),
        weight_decay=params['train']['weight_decay'],
        warmup_steps=params['train']['warmup_steps'],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="mlflow",
        logging_steps=10
    )

    with mlflow.start_run("BERT fine tuning"):
        trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset = train_ds,
            eval_dataset = val_ds,
            optimizers = (optimizer, None),
            processing_class = tokenizer,
            compute_metrics = compute_metrics,
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        )
        trainer.train()
        metrics = trainer.evaluate()

        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(model, "model")

        trainer.save_model(params['train']['model_save_path'])
        tokenizer.save_pretrained(params['train']['tokenizer_path'])

        logger.info(f"Model saved to {params['train']['model_save_path']}")
        logger.info(f"Tokenizer saved to {params['train']['tokenizer_path']}")

if __name__ == "__main__":
    main()