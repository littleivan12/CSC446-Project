import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW  
from datasets import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyabsa import ATEPCCheckpointManager

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 64
BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 3e-5
SENTIMENT_LABELS = ["positive", "neutral", "negative"]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def load_reviews_dataset(file_path):
    df = pd.read_csv(file_path).dropna(subset=['Review', 'Rating'])
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df = df.dropna(subset=['Rating'])

    def map_rating(r):
        return "positive" if r >= 4 else "neutral" if r == 3 else "negative"

    df['Sentiment'] = df['Rating'].apply(map_rating)
    return Dataset.from_dict({"text": df['Review'].tolist(), "label": df['Sentiment'].tolist()}), df


def tokenize(dataset, tokenizer):
    def encode(example):
        tokens = tokenizer(
            example['text'], padding='max_length', truncation=True, max_length=MAX_LENGTH
        )
        tokens["labels"] = SENTIMENT_LABELS.index(example["label"])
        return tokens

    return dataset.map(encode, remove_columns=["text", "label"])

def collate(batch):
    return {
        "input_ids": torch.tensor([x["input_ids"] for x in batch]).to(DEVICE),
        "attention_mask": torch.tensor([x["attention_mask"] for x in batch]).to(DEVICE),
        "labels": torch.tensor([x["labels"] for x in batch]).to(DEVICE),
    }


def train(model, dataloader):
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    losses = []

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        losses.append(total_loss / len(dataloader))
    return losses

def predict(model, tokenizer, texts):
    model.eval()
    predictions = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
        sentiment = SENTIMENT_LABELS[pred]
        predictions.append((text, sentiment))
        print(f"Review: {text}\nPredicted Sentiment: {sentiment}\n")
    return predictions

def run_absa_analysis(reviews):
    print("\nRunning ABSA (Aspect-Based Sentiment Analysis)...\n")
    aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint="english")
    results = aspect_extractor.extract_aspect(inference_source=reviews, print_result=False)
    
    output = []
    for review, result in zip(reviews, results):
        for aspect, sentiment in zip(result["aspect"], result["sentiment"]):
            output.append({
                "Review": review,
                "Aspect": aspect,
                "Aspect Sentiment": sentiment
            })
    absa_df = pd.DataFrame(output)
    absa_df.to_csv("absa_results.csv", index=False)
    print("ABSA results saved to absa_results.csv")
    return absa_df


if __name__ == "__main__":
    dataset, df_reviews = load_reviews_dataset("restaurant_reviews.csv")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized = tokenize(dataset, tokenizer)
    tokenized = tokenized.train_test_split(test_size=0.1)

    train_loader = DataLoader(tokenized["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(DEVICE)
    print(f"Training on: {DEVICE}")
    loss_vals = train(model, train_loader)

    # Plot Loss
    plt.plot(loss_vals)
    plt.title("Loss Over Epochs")
    plt.savefig("loss_fast.png")

  
    test_reviews = [
        "Amazing food and quick service!",
        "The place was okay, nothing great.",
        "Very disappointing experience."
    ]
    sentence_sentiments = predict(model, tokenizer, test_reviews)


    absa_df = run_absa_analysis(df_reviews["Review"].tolist())
