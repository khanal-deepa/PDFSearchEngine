import os
from flask import Flask, request, jsonify, render_template
from indexer import index_files
from search import load_index, search_query

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'}), 400
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No selected files'}), 400

    for file in files:
        if file and file.filename.endswith('.pdf'):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    try:
        index_files(app.config['UPLOAD_FOLDER'])
        return jsonify({'message': 'Files uploaded and indexed successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400

    try:
        bm25, document_chunks, chunk_to_filename, chunk_to_original = load_index()
        results = search_query(query, bm25, document_chunks, chunk_to_original)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)

