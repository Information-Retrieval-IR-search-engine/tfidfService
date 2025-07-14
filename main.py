from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()  # normalize case
    tokens = nltk.word_tokenize(text)  # tokenize
    tokens = [t for t in tokens if t.isalpha()]  # remove punctuation/numbers
    tokens = [t for t in tokens if t not in stop_words]  # remove stopwords
    tokens = [lemmatizer.lemmatize(t) for t in tokens]  # lemmatize
    return ' '.join(tokens)

tfidf2 = joblib.load('tfidf.joblib')
# docs = pd.read_csv('docs_beir.csv')
# docs['original_text'] = docs['text']
# docs['text'] = docs['text'].astype(str).apply(preprocess)
docs = joblib.load('quora_docs.joblib')
# inverted_index = joblib.load('inverted_index.joblib')
        
# docs['original_text'] = docs['text']
# docs['text'] = docs['text'].astype(str).apply(preprocess)
# doc_vectors = tfidf2.transform(docs["text"])
doc_vectors = joblib.load('quora_doc_vectors.joblib')
def retrieve_custom_query2(query_text, top_k=10):
    # Preprocess the query
    query_processed = preprocess(query_text)

    # Vectorize the query
    query_vec = tfidf2.transform([query_processed])

    # Compute cosine similarity
    cosine_scores = cosine_similarity(query_vec, doc_vectors)[0]

    # Get top_k ranked documents
    top_indices = np.argsort(cosine_scores)[::-1][:top_k]

    # Fetch and return the document texts and scores
    results = docs.iloc[top_indices][['doc_id', 'original_text']].copy()
    results['score'] = cosine_scores[top_indices]
    
        # Format as list of dictionaries
    results_ret = [
        {
            "doc_id": int(row["doc_id"]),
            "text": row["original_text"],
            "score": float(row["score"])
        }
        for _, row in results.iterrows()
    ]

    return results_ret




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
@app.post("/search")
def search(query: str = Form(...), algorithm: str = Form(...)):
    results = retrieve_custom_query2(query, top_k=10)
    return {"results": results}


# @app.get("/suggest")
# def suggest(q: str):
#     suggestions = [f"{q} information retrieval", "inverted index", "indexing documents", "intelligent search", "interactive systems"]
#     return suggestions


