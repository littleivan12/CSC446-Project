# word embedding analysis for resuarant reviews
#Performance of sentiment and aspect analysis on restaurant reviews
import os
import ast
import pandas as pd
import torch
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer 

REVIEWS_FILE = "restaurant_reviews.csv"
ASPECTS = ["food", "service", "ambience", "price", "menu"]
SENTIMENTS = ["positive", "neutral", "negative"]

def load_and_preprocess():
    """Load dataset and add aspect/sentiment labels"""
    df = pd.read_csv(REVIEWS_FILE)
    
    # Clean Rating column, convert to numeric and drop invalid entries
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df = df.dropna(subset=['Rating'])
    
    # Add sentiment column based on rating tresholds
    df['Sentiment'] = df['Rating'].apply(
        lambda x: "positive" if x >= 4 else "neutral" if x >= 3 else "negative"
    )

    df['Review'] = (
        df['Review']
        .astype(str)  # Convert everything to string first
        .str.lower()
        .replace('nan', '')  # Handle string 'nan' values
    )

    # Extract aspects with safety checks
    def extract_aspects(text):
        if not isinstance(text, str):  # Double safety check
            return []
        return [aspect for aspect in ASPECTS if aspect in text]
    
    df['Aspects'] = df['Review'].apply(extract_aspects)

    return df


def analyze_embeddings(df):
    """Analyze review embeddings using TF-IDF and visualize sentiment clusters
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Review', 'Aspects', and 'Sentiment' columns
        
    Returns:
        None (saves visualization to 'review_embeddings.png')
    """
    try:
 
        df['Aspects'] = df['Aspects'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

        
        # Vectorize reviews using TF-IDF
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        embeddings = vectorizer.fit_transform(df['Review'])
        
        # Reduce dimensionality for visualization
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings.toarray())
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        colors = {'positive': '#4CAF50',  # Green
                 'neutral': '#2196F3',   # Blue
                 'negative': '#F44336'}  # Red
        
        for sentiment, color in colors.items():
            mask = df['Sentiment'] == sentiment
            plt.scatter(
                reduced_embeddings[mask, 0],
                reduced_embeddings[mask, 1],
                c=color,
                label=sentiment.capitalize(),
                alpha=0.6,
                edgecolors='w',
                linewidths=0.5
            )
        
        plt.title("Restaurant Review Sentiment Clusters", pad=20)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title='Sentiment')
        plt.grid(alpha=0.3)
        

        plt.savefig("review_embeddings.png", bbox_inches='tight', dpi=300)
        plt.close()
        
    except Exception as e:
        print("Error in analyze_embeddings:", str(e))
        raise

    vectorizer = TfidfVectorizer(max_features=1000)
    embeddings = vectorizer.fit_transform(df['Review'])

    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings.toarray())

    plt.figure(figsize=(12, 8))
    colors = {'positive': 'green', 'neutral':'blue', 'negative': 'red'}

    for sentiment in SENTIMENTS: 
        mask = df['Sentiment'] == sentiment
        plt.scatter(
            reduced_embeddings[mask, 0],
            reduced_embeddings[mask, 1],
            c=colors[sentiment],
            label=sentiment,
            alpha=0.5
        )
    plt.title("Review Embeddings by Sentiment(PCA)")
    plt.legend()
    plt.savefig("review_embeddings.png")
    plt.close()
    
def calculate_similarities(df):
    """Compare average embeddings for aspects"""
    #Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=500)
    embeddings = vectorizer.fit_transform(df['Review'])

    print("\nAspect Similarities:")
    aspect_embeddings = {}

    #calculate average embeddings for each aspect
    for aspect in ASPECTS:
        try:
            mask = df['Review'].str.contains(aspect, case=False).values
            if sum(mask) > 0:
                # Convert sparse matrix to array and take mean
                aspect_embeddings[aspect] = np.asarray(embeddings[mask].mean(axis=0)).flatten()
        except Exception as e:
            print(f"Error processing aspect '{aspect}': {str(e)}")
            continue

    #calculate and print pairwise similarities between aspects
    aspects = list(aspect_embeddings.keys())
    for i, aspect1 in enumerate(aspects):
        for aspect2 in aspects[i+1:]:
            try:
                vec1 = np.asarray(aspect_embeddings[aspect1]).reshape(1, -1)
                vec2 = np.asarray(aspect_embeddings[aspect2]).reshape(1, -1)
                
                # Calculate cosine similarity
                sim = cosine_similarity(vec1, vec2)[0][0]
                print(f"{aspect1} → {aspect2}: {sim:.3f}")
                    
            except Exception as e:
                print(f"{aspect1} → {aspect2}: Error ({str(e)})")

if __name__=="__main__":

    df = load_and_preprocess()
    print(f"Loaded {len(df)} reviews with {df['Aspects'].explode().nunique()} aspects")

    analyze_embeddings(df)

    calculate_similarities(df)

    print("\nAnalysis complete! Check review_embeddings.png")
