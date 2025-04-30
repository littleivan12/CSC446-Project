import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import re
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

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

# Load dataset
df = pd.read_csv("restaurant_reviews.csv")

# Preprocessing functions
def clean_text(text):
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

# Preprocessing pipeline
def preprocess_data(df):
    df['cleaned_text'] = df['review'].apply(clean_text)
    df['aspects'] = df['cleaned_text'].apply(aspect_extraction)
    df = df.explode('aspects').reset_index(drop=True)
    df = df[df['aspects'].isin(ASPECT_CATEGORIES)]
    
    # Convert ratings to 3-class sentiment
    df['sentiment'] = pd.cut(df['rating'], bins=[0, 2, 3, 5], 
                            labels=['negative', 'neutral', 'positive'])
    return df

processed_df = preprocess(df)
print(f"Processed dataset size: {len(processed_df)}")

# Split dataset
train_df, test_df = train_test_split(processed_df, test_size=0.2, 
                                   random_state=RANDOM_SEED,
                                   stratify=processed_df['sentiment'])

# BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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
        
        # Format: [CLS] text [SEP] aspect [SEP]
        encoded = tokenizer.encode_plus(
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

# Convert labels to numerical values
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
train_df['label'] = train_df['sentiment'].map(label_map)
test_df['label'] = test_df['sentiment'].map(label_map)

# Create data loaders
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = ABSADataset(
        texts=df.cleaned_text.values,
        aspects=df.aspects.values,
        labels=df.label.values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

train_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
test_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

# Model setup
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3,
    output_attentions=False,
    output_hidden_states=False
).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Training function
def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    losses = []
    correct_predictions = 0
    
    for batch in data_loader:
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
    
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# Evaluation function
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

# Training loop
for epoch in range(EPOCHS):
    train_acc, train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
    test_acc, test_loss = eval_model(model, test_loader, device)
    
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print(f'Train loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}')
    print(f'Test loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}')
    print('-' * 50)

# Performance analysis
def get_predictions(model, data_loader):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return predictions, true_labels

y_pred, y_true = get_predictions(model, test_loader)

# Calculate metrics
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
accuracy = accuracy_score(y_true, y_pred)

print(f'Final Model Performance:')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Save model
torch.save(model.state_dict(), 'absa_bert_model.bin')