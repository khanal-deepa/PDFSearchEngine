
import json
from rank_bm25 import BM25Okapi
from indexer import preprocess_text

INDEX_FILE = 'index.json'

def load_index():
    with open(INDEX_FILE, 'r') as f:
        index_data = json.load(f)
    
    document_chunks = index_data['document_chunks']
    file_to_chunks = index_data['file_to_chunks']
    chunk_to_filename = index_data['chunk_to_filename']
    chunk_to_original = index_data['chunk_to_original']
    bm25_idf = index_data['bm25_idf']
    
    bm25 = BM25Okapi(document_chunks)
    bm25.idf = bm25_idf
    
    return bm25, document_chunks, chunk_to_filename, chunk_to_original

def search_query(query, bm25, document_chunks, chunk_to_original, top_n=5):
    query_tokens = preprocess_text(query)
    scores = bm25.get_scores(query_tokens)
    top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    
    results = []
    for index in top_n_indices:
        chunk_key = ' '.join(document_chunks[index])
        original_chunk = chunk_to_original[chunk_key]
        results.append(original_chunk)
    
    return results

if __name__ == "__main__":
    query = "organizational goal"
    bm25, document_chunks, chunk_to_filename, chunk_to_original = load_index()
    results = search_query(query, bm25, document_chunks, chunk_to_original)
    for result in results:
        print(f"{result}\n")
