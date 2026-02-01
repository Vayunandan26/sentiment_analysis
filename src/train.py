from datasets import Dataset
import logging
import os
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/Users/vayunandan/Documents/BERT/sentiment_analysis/data.log"),
        logging.StreamHandler()
    ]
)

with open('data.log', 'w'):
    pass

logger = logging.getLogger(__name__)

# 1. Disable WandB to prevent login hangs
os.environ["WANDB_MODE"] = "disabled"

# 4. Model Setup
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

def tokenize_function(examples):
    return tokenizer(examples['tweet'], padding='max_length', truncation=True, max_length=64)

train_df = 'train_augmented.csv'
val_df = 'val.csv'

train_df['tweet'] = train_df['tweet'].fillna('').astype(str)
val_df['tweet'] = val_df['tweet'].fillna('').astype(str)

train_dataset = Dataset.from_pandas(train_df[['tweet', 'labels']])
val_dataset = Dataset.from_pandas(val_df[['tweet', 'labels']])

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# 5. Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,              # Increased slightly
    per_device_train_batch_size=16,  # Larger batches provide more stable gradients
    learning_rate=1e-5,              # LOWER learning rate (Crucial!)
    weight_decay=0.01,
    lr_scheduler_type="linear",      # Helps the model settle
    warmup_steps=100,                # Slowly builds up the learning rate
    eval_strategy="epoch",           # Watch it epoch by epoch
    save_strategy="epoch",
    load_best_model_at_end=True      # Keeps the best version automatically
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics 
)

trainer.train()