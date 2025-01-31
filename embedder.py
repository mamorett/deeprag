from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
import logging
import psutil
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores.chroma import Chroma
import chromadb
from ebooklib import epub
from bs4 import BeautifulSoup
import torch
from transformers import AutoModel, AutoTokenizer


class RAGPipeline:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", max_memory_gb: float = 3.0):
        self.setup_logging()
        self.check_system_memory(max_memory_gb)
        # Load the language model (LLM)
        self.llm = OllamaLLM(model="deepseek-r1:8b")
        
        # Initialize embeddings using a lightweight model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}  # Use CPU for efficiency
        )

        self.model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.model.eval()        

        # Initialize Chroma client - Fixed configuration for remote server
        self.chroma_client = chromadb.HttpClient(
            host="localhost",
            port=18000
        )
        # Initialize Chroma collection
        self.collection_name = "oreilly"
        # Delete the old collection
        # self.chroma_client.delete_collection(self.collection_name)

        # Create a new collection with the correct dimension
        
        self.collection = self.chroma_client.get_or_create_collection(
            self.collection_name,
            metadata={"dimensionality": 768}  # Explicitly define the correct embedding size
        )



    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def check_system_memory(self, max_memory_gb: float):
        available_memory = psutil.virtual_memory().available / (1024 ** 3)
        self.logger.info(f"Available system memory: {available_memory:.1f} GB")
        if available_memory < max_memory_gb:
            self.logger.warning("Memory is below recommended threshold.")

    # Function to extract text from an EPUB file
    def extract_text_from_epub(self, epub_path):
        book = epub.read_epub(epub_path)
        text_content = []
        for item in book.get_items():
            if isinstance(item, epub.EpubHtml):
                soup = BeautifulSoup(item.content, 'html.parser')
                text_content.append(soup.get_text())
        return " ".join(text_content)
    
    # Function to embed text using the model
    def embed_text(self, text):
        tokenized = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.model(**tokenized)
            embeddings = model_output[0][:, 0]  # CLS pooling
        return torch.nn.functional.normalize(embeddings, dim=1).squeeze().tolist()    

    def load_and_split_documents(self, file_path: str) -> List[Document]:
        # Step 1 - load and split documents
        # Process EPUB files in the ./epubs directory
        epub_dir = "./epubs"
        for filename in os.listdir(file_path):
            if filename.endswith(".epub"):
                epub_path = os.path.join(epub_dir, filename)
                print(f"Processing: {epub_path}")
                
                # Extract text and create embeddings
                text = self.extract_text_from_epub(epub_path)
                embedding = self.embed_text(text)
                
                # Generate a unique ID from the filename (without the .epub extension)
                doc_id = os.path.splitext(filename)[0]
                
                # Add to Chroma database with the unique ID
                self.collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    metadatas=[{"filename": filename}],
                    documents=[text],
                )


    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Create a vector store from the provided documents using Chroma.
        
        Args:
            documents: List of Document objects to be added to the vector store
            
        Returns:
            Chroma: Initialized vector store containing the document embeddings
        """
        # Create and initialize Chroma vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            client=self.chroma_client
        )
        
        self.logger.info(f"Created vector store with {len(documents)} documents")
        return vectorstore

def main():
    rag = RAGPipeline(model_name="deepseek-r1:8b", max_memory_gb=3.0)
    
    # documents = rag.load_and_split_documents("data/knowledge.txt")
    documents = rag.load_and_split_documents("./epubs")

    vectorstore = rag.create_vectorstore(documents)

if __name__ == "__main__":
    main()   