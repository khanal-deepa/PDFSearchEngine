import os
import json
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string
from rank_bm25 import BM25Okapi

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token not in string.punctuation]
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def read_pdf(file_path):
    pdf_document = fitz.open(file_path)
    text = ''
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text += page.get_text()
    return text

def divide_text_into_chunks(text):
    lines = text.split('\n')
    chunks = []
    chunk = ""
    for line in lines:
        if line.strip() and line.strip()[0].isupper():  # A simple heuristic to detect headings
            if chunk:
                chunks.append(chunk.strip())
                chunk = ""
        chunk += line + "\n"
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def index_files(directory_path):
    document_chunks = []
    file_to_chunks = {}
    chunk_to_filename = {}
    chunk_to_original = {}
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            content = read_pdf(file_path)
            if not content:
                print(f"Warning: {filename} is empty after text extraction and will be skipped.")
                continue
            
            chunks = divide_text_into_chunks(content)
            file_to_chunks[filename] = chunks
            for chunk in chunks:
                preprocessed_chunk = preprocess_text(chunk)
                document_chunks.append(preprocessed_chunk)
                chunk_key = ' '.join(preprocessed_chunk)
                chunk_to_filename[chunk_key] = filename
                chunk_to_original[chunk_key] = chunk
    
    if not document_chunks:
        raise ValueError("All PDF files are empty after preprocessing")
    
    bm25 = BM25Okapi(document_chunks)
    
    # Save index to a file
    index_data = {
        'document_chunks': document_chunks,
        'file_to_chunks': file_to_chunks,
        'chunk_to_filename': chunk_to_filename,
        'chunk_to_original': chunk_to_original,
        'bm25_idf': bm25.idf,  # Directly save the dictionary
    }
    with open('index.json', 'w') as f:
        json.dump(index_data, f)
    
    print('Index file saved as index.json')

if __name__ == "__main__":
    directory_path = 'uploads'  # Replace with your directory containing PDF files
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    index_files(directory_path)

