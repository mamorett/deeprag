from flask import Flask, render_template, request, jsonify
from typing import Optional
import threading
import queue
from RAGPipeline import RAGPipeline

app = Flask(__name__)

# Initialize RAG pipeline globally
rag = RAGPipeline(model_name="deepseek-r1:8b", max_memory_gb=3.0)
chain = rag.setup_rag_chain(rag.collection)

# Create templates directory and add this HTML file
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        response = rag.query(chain, question)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5500)
