
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
import psycopg2
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# === DATABASE CONFIG ===
def get_docs_from_postgres(dataset):
    conn = psycopg2.connect(
        host="localhost",  # update as needed
        dbname="ir_db",
        user="postgres",
        password="your_password"
    )
    query = f"SELECT doc_id, text FROM {dataset}_documents;"
    df = pd.read_sql(query, conn)
    conn.close()

    df['original_text'] = df['text']
    df['text'] = df['text'].astype(str).apply(preprocess)
    return df

# === GLOBAL MODELS & VECTORS (Lazy loading) ===
loaded = {}

def load_resources(dataset):
    if dataset in loaded:
        return loaded[dataset]

    # Load docs
    docs = get_docs_from_postgres(dataset)

    # Load tfidf vectorizer
    tfidf_vectorizer = joblib.load(f"{dataset}_tfidf.joblib")

    # Load doc vectors
    doc_vectors = joblib.load(f"{dataset}_doc_vectors.joblib")

    # Build inverted index
    inverted_index = defaultdict(set)
    for idx, row in docs.iterrows():
        for term in row['text'].split():
            inverted_index[term].add(row['doc_id'])

    loaded[dataset] = {
        "docs": docs,
        "tfidf": tfidf_vectorizer,
        "vectors": doc_vectors,
        "inverted_index": inverted_index
    }
    return loaded[dataset]

# === MAIN RETRIEVAL FUNCTION ===
def retrieve_custom_query(query_text, dataset, top_k=10):
    resources = load_resources(dataset)
    docs = resources["docs"]
    tfidf = resources["tfidf"]
    doc_vectors = resources["vectors"]

    query_processed = preprocess(query_text)
    query_vec = tfidf.transform([query_processed])
    cosine_scores = cosine_similarity(query_vec, doc_vectors)[0]
    top_indices = np.argsort(cosine_scores)[::-1][:top_k]

    results = docs.iloc[top_indices][['doc_id', 'original_text']].copy()
    results['score'] = cosine_scores[top_indices]

    return [
        {
            "doc_id": int(row["doc_id"]),
            "text": row["original_text"],
            "score": float(row["score"])
        }
        for _, row in results.iterrows()
    ]


# === FASTAPI APP ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/search")
def search(query: str = Form(...), algorithm: str = Form(...), dataset: str = Form(...)):
    if dataset not in ["quora", "antique"]:
        return {"error": "Unsupported dataset"}

    results = retrieve_custom_query(query, dataset, top_k=10)
    return {"results": results}
