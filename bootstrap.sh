#!/bin/bash
sudo python3 -m pip install \
    "numpy<2.0" \
    rank_bm25 \
    sentence-transformers==2.2.2 \
    transformers==4.28.1 \
    huggingface-hub==0.14.1 \
    faiss-cpu \
    nltk \
    stanza \
    typing_extensions==4.5.0 \
  --no-cache-dir

# Prepare a local cache
mkdir -p /mnt/hf_cache
chown hadoop:hadoop /mnt/hf_cache

# Pre-download the model into /mnt/hf_cache
python3 - <<'PYCODE'
import os
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/mnt/hf_cache"
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
PYCODE

# Push the cache to S3 for reuse later
aws s3 sync /mnt/hf_cache s3://nina-rag-project/hf_cache/