import os
import time
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    pipeline
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Configuration
REVIEWS_FILE = "restaurant_reviews.csv"
MODEL_NAME = "gpt2"
TRAIN_EPOCHS = 10
BATCH_SIZE = 32
MAX_LENGTH = 180
OUTPUT_DIR = "./review_model"
GRADIENT_ACCUMULATION = 1
SENTIMENT_TOKENS = ["[POSITIVE]", "[NEUTRAL]", "[NEGATIVE]"]

def load_reviews_dataset(file_path):
    # Load and clean the dataset
    df = pd.read_csv(file_path)
    
    # Basic cleaning
    df = df.dropna(subset=['Review', 'Rating'])
    df['Review'] = df['Review'].str.strip()
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df = df.dropna(subset=['Rating'])
    
    # Convert ratings to sentiment categories
    def rating_to_sentiment(rating):
        if rating >= 4:
            return SENTIMENT_TOKENS[0]
        elif rating >= 3:
            return SENTIMENT_TOKENS[1]
        else:
            return SENTIMENT_TOKENS[2]
    
    df['Sentiment'] = df['Rating'].apply(rating_to_sentiment)
    df['Text'] = df['Sentiment'] + " " + df['Review']
    
    return Dataset.from_dict({"text": df['Text'].tolist()})


def tokenize_reviews(dataset):
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({'additional_special_tokens': SENTIMENT_TOKENS})
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset.train_test_split(test_size=0.1), tokenizer

def train_model(train_dataset, tokenizer):
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))  # Important for new tokens
    
    print(f"\nTraining device: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        save_steps=2000,
        save_total_limit=2,
        fp16=True,
        logging_dir='./logs',
        dataloader_num_workers=4,
        remove_unused_columns=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    start_time = time.time()
    trainer.train()
    return model, tokenizer, time.time() - start_time, trainer

def generate_reviews(model, tokenizer, base_prompt):
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    generated = {}
    for sentiment in SENTIMENT_TOKENS:
        full_prompt = f"{sentiment} {base_prompt}"
        result = generator(
            full_prompt,
            max_length=MAX_LENGTH,
            num_return_sequences=1,
            temperature=0.7,
            repetition_penalty=1.1,
            do_sample=True
        )
        
        generated[sentiment] = result[0]['generated_text'].replace(full_prompt, "").strip()
    return generated

def plot_training_loss(log_history):
    losses = [log['loss'] for log in log_history if 'loss' in log]
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-o')
    plt.title("Training Loss Progress")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("training_loss.png")
    plt.close()

def print_hardware_specs(train_time):
    print("\nHardware Specifications:")
    print(f"CPU: {psutil.cpu_percent()}% utilization")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Training Time: {train_time / 60:.1f} minutes")

if __name__ == "__main__":
    dataset = load_reviews_dataset(REVIEWS_FILE)
    split_dataset, tokenizer = tokenize_reviews(dataset)
    
    model, tokenizer, train_time, trainer = train_model(split_dataset['train'], tokenizer)
    
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    test_prompts = [
        "The food was",
        "The service was",
        "The ambience was",
        "I would recommend this place because",
        "The menu offered"
    ]
    
    print("\nGenerated Reviews:")
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        reviews = generate_reviews(model, tokenizer, prompt)
        for sentiment, text in reviews.items():
            print(f"{sentiment}: {text}")
    
    print_hardware_specs(train_time)
    plot_training_loss(trainer.state.log_history)