"""
Common utility functions for text processing, file I/O, and data preprocessing.
"""

import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess

def load_and_preprocess_docs(file_path, date_filter=None):
    """Load and preprocess documents from CSV file."""
    docs = pd.read_csv(file_path, encoding="utf-8")
    
    if date_filter:
        docs['Date'] = docs['Date'].astype(int)
        docs = docs[docs['Date'] <= date_filter]
    
    docs = docs['Speech'].astype(str)
    docs = docs.reset_index(drop=True)
    return docs

def preprocess_text(text, remove_numbers=True, min_length=1):
    """Preprocess a single text document."""
    # Handle NaN/None values and convert to string
    if pd.isna(text) or text is None:
        return []
        
    # Convert to string if not already
    text = str(text)
    
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())
    
    if remove_numbers:
        tokens = [token for token in tokens if not token.isnumeric()]
    
    if min_length > 1:
        tokens = [token for token in tokens if len(token) > min_length]
    
    return tokens

def remove_custom_stopwords(texts, additional_stopwords=None):
    """Remove stopwords including custom ones."""
    stop_words = set(stopwords.words('english'))
    if additional_stopwords:
        stop_words.update(additional_stopwords)
    
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]

def lemmatize_texts(texts):
    """Lemmatize a list of tokenized texts."""
    lemmatizer = WordNetLemmatizer()
    return [[lemmatizer.lemmatize(token) for token in doc] for doc in texts]

def normalize_matrix_column(matrix):
    """Normalize matrix columns to sum to 1."""
    return matrix / matrix.sum(axis=0, keepdims=True)

def get_chair_for_date(date, chair_data):
    """Get FOMC chair for a given date."""
    for start_date, end_date, chair in chair_data:
        if start_date <= date <= end_date:
            return chair
    return None 