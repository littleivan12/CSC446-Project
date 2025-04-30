# CSC446 Final Project
### Aspect-Based Sentiment Analysis for Restaurant Reviews
Authors: Natalie Hildreth, Ivan Ramos Candelero,and Jennifer Galicia-Torres 

## Description
This project implements Aspect-Based Sentiment Classification (ABSA) for restaurant reviews using BERT and spaCy. Our restaurant review dataset can be found at https://www.kaggle.com/datasets/joebeachcapital/restaurant-reviews. This system identifies sentiment polarity (positive/neutral/negative) for specific aspects like food, service, and ambiance within customer reviews. Our implementation focuses on fine-grained sentiment analysis at the aspect level rather than document-level sentiment.

## Key Features:
- Aspect extraction using dependency parsing
- BERT-based sentiment classification
- Performance evaluation (Accuracy, F1, Precision, Recall)
- GPU acceleration support

## Installation
1. Clone Repository
2. Download Required Imports: 
- Python 3.8+
- pip install pandas numpy spacy sklearn transformers torch
- python -m spacy download en_core_web_sm