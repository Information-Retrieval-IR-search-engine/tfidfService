from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.orm import sessionmaker
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Preprocessing setup ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# --- TF-IDF & inverted index loading ---
tfidf = {}
inverted_index = {}
tfidf['quora'] = joblib.load('tfidf.joblib')
inverted_index['quora'] = joblib.load('inverted_index.joblib')  # {term: set(doc_ids)}
tfidf['antique'] = joblib.load('antique/antique_tfidf.joblib')
inverted_index['antique'] = joblib.load('antique/antique_inverted_index.joblib')

# --- PostgreSQL Connection ---
DATABASE_URL = "postgresql://postgres:123456789@localhost/ir_db"  # change credentials
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def fetch_documents_by_ids(dataset, doc_ids):
    session = SessionLocal()
    try:
        placeholders = ','.join([':id{}'.format(i) for i in range(len(doc_ids))])
        sql = f"SELECT doc_id, text FROM {dataset} WHERE doc_id IN ({placeholders})"
        params = {f'id{i}': str(doc_id) for i, doc_id in enumerate(doc_ids)}
        result = session.execute(sql_text(sql), params).fetchall()
        return [{'doc_id': row[0], 'text': row[1]} for row in result]
    finally:
        session.close()

# --- Inverted Index Retrieval Function ---
def retrieve_with_inverted_index(dataset, query_text, top_k=10):
    query_processed = preprocess(query_text)
    query_terms = query_processed.split()

    # Find matching document IDs
    doc_scores = {}
    for term in query_terms:
        matching_ids = inverted_index[dataset].get(term, set())
        for doc_id in matching_ids:
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1

    if not doc_scores:
        return []

    # Sort by score and take top_k doc_ids
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    doc_ids = [doc_id for doc_id, _ in sorted_docs]

    # Fetch document texts from the database
    retrieved_docs = fetch_documents_by_ids(dataset, doc_ids)

    # Vectorize documents and query
    doc_texts = [preprocess(doc['text']) for doc in retrieved_docs]
    doc_vectors = tfidf[dataset].transform(doc_texts)
    query_vec = tfidf[dataset].transform([query_processed])
    cosine_scores = cosine_similarity(query_vec, doc_vectors)[0]

    # Build result list
    results = []
    for i, doc in enumerate(retrieved_docs):
        results.append({
            "doc_id": doc['doc_id'],
            "text": doc['text'],
            "score": float(cosine_scores[i])
        })

    # Sort by cosine score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

# --- FastAPI setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/search")
def search(query: str = Form(...), dataset: str = Form(...)):
    results = retrieve_with_inverted_index(dataset, query, top_k=10)
    return {"results": results}
