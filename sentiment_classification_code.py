import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import re
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

# Verify GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Configuration
RANDOM_SEED = 42
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
ASPECT_CATEGORIES = ['food', 'service', 'ambiance', 'price', 'quality']

# Preprocessing functions
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower().strip()
    return text

def aspect_extraction(text):
    doc = nlp(text)
    aspects = []
    for chunk in doc.noun_chunks:
        if chunk.root.dep_ in ('nsubj', 'dobj', 'attr') and chunk.text.lower() not in STOP_WORDS:
            aspects.append(chunk.text.lower())
    return list(set(aspects))

def preprocess_data(df):
    df = df.copy()  # Fix SettingWithCopyWarning
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df = df.dropna(subset=['Review', 'Rating'])
    df['cleaned_text'] = df['Review'].apply(clean_text)
    df['aspects'] = df['cleaned_text'].apply(aspect_extraction)
    df = df.explode('aspects').reset_index(drop=True)
    df = df[df['aspects'].isin(ASPECT_CATEGORIES)]
    df['sentiment'] = pd.cut(df['Rating'], bins=[0, 2, 3, 5], 
                            labels=['negative', 'neutral', 'positive'])
    return df

class ABSADataset(Dataset):
    def __init__(self, texts, aspects, labels, tokenizer, max_len):
        self.texts = texts
        self.aspects = aspects
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        aspect = str(self.aspects[idx])
        label = self.labels[idx]
        
        encoded = self.tokenizer.encode_plus(
            text,
            aspect,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = ABSADataset(
        texts=df.cleaned_text.values,
        aspects=df.aspects.values,
        labels=df.label.values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

def train_epoch(model, data_loader, optimizer, device, scheduler, epoch):
    model.train()
    losses = []
    correct_predictions = 0
    
    progress_bar = tqdm(data_loader, desc=f'Epoch {epoch + 1}', leave=False)
    start_time = time.time()
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        elapsed = time.time() - start_time
        avg_time = elapsed / len(losses)
        remaining = avg_time * (len(data_loader) - len(losses))
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'elapsed': f'{elapsed:.1f}s',
            'remaining': f'{remaining:.1f}s'
        })
    
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

if __name__ == '__main__':
    # Load and process data
    try:
        df = pd.read_csv("restaurant_reviews.csv")
        print("Dataset loaded successfully")
    except FileNotFoundError:
        print("Error: File not found")
        exit()

    df = preprocess_data(df)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, stratify=df['sentiment'])

    # Prepare data loaders
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    train_df['label'] = train_df['sentiment'].map(label_map)
    test_df['label'] = test_df['sentiment'].map(label_map)

    train_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
    test_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

    # Initialize model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3,
        output_attentions=False,
        output_hidden_states=False
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training loop
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, device, scheduler, epoch)
        test_acc, test_loss = eval_model(model, test_loader, device)
        
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        print(f'Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}')

    # Save model
    torch.save(model.state_dict(), 'absa_bert_model.bin')
    print("\nTraining complete. Model saved.")