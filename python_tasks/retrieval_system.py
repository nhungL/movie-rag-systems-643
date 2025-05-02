import os
NLTK_DATA = "/mnt/hf_cache/nltk_data"
os.makedirs(NLTK_DATA, exist_ok=True)
os.environ["STANZA_RESOURCES_DIR"] = "/mnt/stanza_resources"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/hf_cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/mnt/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
import numpy as np
import faiss
import stanza
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from pyspark.sql import SparkSession

import nltk
nltk.data.path.insert(0, NLTK_DATA)
for pkg in ("stopwords", "punkt_tab", "averaged_perceptron_tagger"):
    nltk.download(pkg, download_dir=NLTK_DATA, quiet=True)

from nltk.corpus    import stopwords
from nltk.tokenize  import word_tokenize

stopwords_set = set(stopwords.words("english"))

EMBED_PATH = "s3a://nina-rag-project/processed_data/movie_embeddings/part-*"
CLEANED_PATH = "s3a://nina-rag-project/processed_data/movie_cleanedtexts/part-*"

PY3 = "/usr/bin/python3"
spark = SparkSession.builder \
    .appName("Hybrid RAG Retrieval") \
    .config("spark.pyspark.python", PY3) \
    .config("spark.pyspark.driver.python", PY3) \
    .config("spark.executorEnv.PYSPARK_PYTHON", PY3) \
    .config("spark.executorEnv.PYSPARK_DRIVER_PYTHON", PY3) \
    .getOrCreate()

sc = spark.sparkContext

# Load Embeddings
rdd = sc.textFile(EMBED_PATH).cache()
kv_rdd = rdd.map(lambda line: line.split("\t", 1)) \
             .filter(lambda parts: len(parts) == 2) \
             .map(lambda parts: (parts[0], np.fromstring(parts[1], sep="," , dtype="float32")))
data = kv_rdd.collect()
titles, vectors = zip(*data)
embeddings = np.vstack(vectors)

# Dense Retrieval using Faiss
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "movie_rag_index.faiss")

with open("movie_keys.json", "w") as f:
    json.dump(titles, f)
    
text_rdd = sc.textFile(CLEANED_PATH).cache()
text_kv  = text_rdd.map(lambda l: l.split("\t",1)).filter(lambda p: len(p)==2).collect()
title_to_text = dict(text_kv)

# Sparse Retrieval using BM25
plots_filtered = list(title_to_text.values())
tokenized_corpus = [word_tokenize(doc.lower()) for doc in plots_filtered]
bm25 = BM25Okapi(tokenized_corpus)

def sparse_retrieve(query, k=5):
    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    top_k_indices = np.argsort(scores)[::-1][:k]
    return [plots_filtered[i] for i in top_k_indices], top_k_indices

# Hybrid Retrieval
stanza.download("en")
nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma")
genre_mapping = {
    "sci-fi": "science fiction",
    "scifi": "science fiction",
    "sci fi": "science fiction",
    "science fiction": "science fiction",
    "romcom": "romantic comedy",
    "bio": "biography",
    "doc": "documentary",
    "anime": "animated",
    "thriller": "thriller",
    "horror": "horror",
    "action": "action",
    "comedy": "comedy",
    "drama": "drama",
    "mystery": "mystery",
    "fantasy": "fantasy",
    "adventure": "adventure",
    "crime": "crime",
    "war": "war",
    "romance": "romance",
    "family": "family",
    "history": "history",
}

def detect_genres(query):
    query_lower = query.lower()
    detected = []
    for variant, canonical in genre_mapping.items():
        if variant in query_lower:
            detected.append(canonical)
    return list(set(detected))

TARGET_POS = {"NOUN", "PROPN", "ADJ", "NUM"}
def extract_keywords(text):
    doc = nlp(text)

    common_movie_words = {
        "movie", "film", "story", "show", "watch", "scene", "actor", "actress",
        "director", "character", "plot", "cast", "version", "remake", 
        "see", "watching", "episode", "cinema"
    }
    
    all_stopwords = stopwords_set.union(common_movie_words)

    keywords = []
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.upos in TARGET_POS and word.lemma.lower() not in all_stopwords:
                keywords.append(word.lemma.lower())

    return keywords

model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder="/mnt/hf_cache")
def hybrid_retrieve(query, k=5, dense_weight=0.5, sparse_weight=0.5, boost_genre=True, boost_title=True):
    detected_genre = detect_genres(query)
    print("Detected genres from query:", detected_genre)

    keywords = extract_keywords(query)
    print("Extracted keywords from query:", keywords)
    processed_query = " ".join(keywords)

    # Encode query with SBERT
    dense_vector = model.encode([processed_query]).astype("float32")
    D, I = index.search(dense_vector, k=10)
    dense_indices = I[0]

    # Sparse retrieval (BM25)
    _, sparse_indices = sparse_retrieve(processed_query, k=10)

    # Combine scores from both retrievals
    combined_scores = {}
    for rank, idx in enumerate(dense_indices):
        combined_scores[idx] = combined_scores.get(idx, 0) + dense_weight * (1 / (1 + rank))
    for rank, idx in enumerate(sparse_indices):
        combined_scores[idx] = combined_scores.get(idx, 0) + sparse_weight * (1 / (1 + rank))

    # Boost scores if genres match
    if boost_genre and detected_genre:
        for i, title in enumerate(titles):
            genre_text = title_to_text.get(title, "").lower()
            if "genre: unknown" in genre_text:
                combined_scores[i] = combined_scores.get(i, 0) - 0.75
            for g in detected_genre:
                if g in genre_text:
                    combined_scores[i] = combined_scores.get(i, 0) + 1.0
                    break

    # Boost if movie title appears in query
    if boost_title:
        query_lower = query.lower()
        for i, title in enumerate(titles):
            if title.lower().split("_")[0] in query_lower:
                combined_scores[i] = combined_scores.get(i, 0) + 2.0

    # Sort final scores
    sorted_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)

    # Filter to keep only movies with matching or unknown genre
    if detected_genre:
        genre_filtered_indices = []
        for i in sorted_indices:
            genre_section = title_to_text.get(titles[i], "").lower()
            if "genre: unknown" in genre_section or any(g in genre_section for g in detected_genre):
                genre_filtered_indices.append(i)
        if genre_filtered_indices:
            sorted_indices = genre_filtered_indices[:k]

    # Final top results
    top_titles = [titles[i] for i in sorted_indices[:k]]
    top_docs = [title_to_text[t] for t in top_titles]

    return top_docs, top_titles

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', required=True)
    parser.add_argument('--topk', type=int, default=5)
    args = parser.parse_args()
    
    top_docs, top_titles = hybrid_retrieve(args.query, k=args.topk)
    for doc, title in zip(top_docs, top_titles):
        print(f"**{title}**\n{doc[:300]}...\n")
    
    results = {"query": args.query, "titles": top_titles, "docs": top_docs}
    with open("retrieval_results.json", "w") as f:
        json.dump(results, f)
    
    spark.stop()