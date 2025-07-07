from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
import logging
import psutil
import os
import oracledb
from ebooklib import epub
from bs4 import BeautifulSoup
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import array
import json
from pathlib import Path
import hashlib
import socket
from dotenv import load_dotenv


class RAGPipeline:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", max_memory_gb: float = 3.0):
        # Load environment variables from .env file
        load_dotenv()
        
        self.setup_logging()
        self.check_system_memory(max_memory_gb)
        
        # Check for GPU availability
        self.device = self.setup_device()
        
        # Load the language model (LLM)
        self.llm = OllamaLLM(model="mistral-nemo:12b")
        
        # Initialize embeddings using GPU if available (simplified)
        model_kwargs = {'device': self.device}
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs=model_kwargs
        )

        # Load model and tokenizer with GPU support
        self.model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        
        # Move model to GPU if available
        self.model = self.model.to(self.device)
        if self.device == 'cuda':
            # Convert to half precision after moving to GPU
            self.model = self.model.half()
        
        self.model.eval()        

        # Initialize Oracle connection
        self.setup_oracle_connection()
        self.table_name = "DOCUMENT_VECTORS"
        self.setup_oracle_tables()

    def setup_device(self):
        """Setup device (GPU/CPU) for embeddings"""
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.info(f"GPU detected: {gpu_name}")
            self.logger.info(f"GPU memory: {gpu_memory:.1f} GB")
            self.logger.info("Using GPU for embeddings")
        else:
            device = 'cpu'
            self.logger.info("No GPU detected, using CPU for embeddings")
        
        return device

    def setup_oracle_connection(self):
        """Setup Oracle connection using DSN string from environment"""
        try:
            # Get Oracle connection parameters from environment variables
            user = os.getenv('ORACLE_USER')
            password = os.getenv('ORACLE_PASSWORD')
            dsn = os.getenv('ORACLE_DSN')
            
            # Validate required environment variables
            required_vars = ['ORACLE_USER', 'ORACLE_PASSWORD', 'ORACLE_DSN']
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
            self.logger.info("Connecting to Oracle using provided DSN")
            
            # Test single connection first
            self.logger.info("Testing connection...")
            test_connection = oracledb.connect(
                user=user,
                password=password,
                dsn=dsn
            )
            
            # Test with a simple query
            cursor = test_connection.cursor()
            cursor.execute("SELECT 'Connection successful' FROM DUAL")
            result = cursor.fetchone()
            self.logger.info(f"Connection test result: {result[0]}")
            cursor.close()
            test_connection.close()
            
            # Create connection pool
            self.connection_pool = oracledb.create_pool(
                user=user,
                password=password,
                dsn=dsn,
                min=1,
                max=5,
                increment=1
            )
            
            self.logger.info("Successfully connected to Oracle database")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Oracle: {e}")
            self.logger.error("Please ensure your .env file contains:")
            self.logger.error("- ORACLE_USER (your database username)")
            self.logger.error("- ORACLE_PASSWORD (your database password)")
            self.logger.error("- ORACLE_DSN (your full DSN connection string)")
            raise

    def setup_oracle_tables(self):
        """Create the document vectors table with specified schema"""
        try:
            with self.connection_pool.acquire() as connection:
                cursor = connection.cursor()
                
                # Test the connection
                cursor.execute("SELECT 'Table setup starting' FROM DUAL")
                result = cursor.fetchone()
                self.logger.info(f"Connection test: {result[0]}")
                
                # Check if table exists
                cursor.execute("""
                    SELECT COUNT(*) FROM USER_TABLES WHERE TABLE_NAME = :1
                """, [self.table_name])
                
                table_exists = cursor.fetchone()[0] > 0
                
                if not table_exists:
                    create_table_sql = f"""
                    CREATE TABLE {self.table_name} (
                        DOCID VARCHAR2(255) PRIMARY KEY,
                        BODY CLOB NOT NULL,
                        VECTOR VECTOR(768, FLOAT32) NOT NULL,
                        CHUNKID VARCHAR2(255),
                        URL VARCHAR2(2000),
                        TITLE VARCHAR2(1000),
                        PAGE_NUMBERS VARCHAR2(500),
                        CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                    
                    cursor.execute(create_table_sql)
                    connection.commit()
                    self.logger.info(f"Table {self.table_name} created successfully")
                else:
                    self.logger.info(f"Table {self.table_name} already exists")
                
                # Check if vector index exists
                cursor.execute("""
                    SELECT COUNT(*) FROM USER_INDEXES WHERE INDEX_NAME = :1
                """, [f"{self.table_name}_VECTOR_IDX"])
                
                index_exists = cursor.fetchone()[0] > 0
                
                if not index_exists:
                    create_index_sql = f"""
                    CREATE VECTOR INDEX {self.table_name}_VECTOR_IDX 
                    ON {self.table_name} (VECTOR) 
                    ORGANIZATION NEIGHBOR PARTITIONS
                    WITH DISTANCE DOT
                    """
                    
                    try:
                        cursor.execute(create_index_sql)
                        connection.commit()
                        self.logger.info("Vector index created successfully")
                    except Exception as e:
                        self.logger.warning(f"Could not create vector index: {e}")
                else:
                    self.logger.info("Vector index already exists")
                
        except Exception as e:
            self.logger.error(f"Failed to setup Oracle tables: {e}")
            raise

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def check_system_memory(self, max_memory_gb: float):
        available_memory = psutil.virtual_memory().available / (1024 ** 3)
        self.logger.info(f"Available system memory: {available_memory:.1f} GB")
        if available_memory < max_memory_gb:
            self.logger.warning("Memory is below recommended threshold.")

    def extract_text_from_epub(self, epub_path):
        """Extract text from an EPUB file"""
        book = epub.read_epub(epub_path)
        text_content = []
        for item in book.get_items():
            if isinstance(item, epub.EpubHtml):
                soup = BeautifulSoup(item.content, 'html.parser')
                text_content.append(soup.get_text())
        return " ".join(text_content)
    
    def embed_text(self, text):
        """Function to embed text using the model with GPU support"""
        tokenized = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=512)
        
        # Move tokenized inputs to the same device as model
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        
        with torch.no_grad():
            model_output = self.model(**tokenized)
            embeddings = model_output[0][:, 0]
            
        # Normalize and convert to CPU for further processing
        normalized = torch.nn.functional.normalize(embeddings, dim=1).squeeze()
        
        # Convert to float32 and move to CPU before converting to list
        if self.device == 'cuda':
            normalized = normalized.float().cpu()
        
        return normalized.tolist()

    def embed_texts_batch(self, texts, batch_size=32):
        """Batch embedding for better GPU utilization"""
        all_embeddings = []
        
        self.logger.info(f"Processing {len(texts)} texts in batches of {batch_size}")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Show progress
            if i % (batch_size * 10) == 0:
                self.logger.info(f"  Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            tokenized = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
                max_length=512
            )
            
            # Move to device
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            
            with torch.no_grad():
                model_output = self.model(**tokenized)
                embeddings = model_output[0][:, 0]  # Take [CLS] token
                
            # Normalize
            normalized = torch.nn.functional.normalize(embeddings, dim=1)
            
            # Convert to CPU and float32
            if self.device == 'cuda':
                normalized = normalized.float().cpu()
            
            # Convert to list and add to results
            for embedding in normalized:
                all_embeddings.append(embedding.tolist())
            
            # Clear GPU cache periodically
            if self.device == 'cuda' and i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
        
        return all_embeddings

    def generate_doc_id(self, filename: str, chunk_index: int = None) -> str:
        """Generate a unique document ID"""
        base_name = os.path.splitext(filename)[0]
        if chunk_index is not None:
            return f"{base_name}_chunk_{chunk_index:04d}"
        else:
            return base_name

    def generate_chunk_id(self, filename: str, chunk_index: int) -> str:
        """Generate a chunk ID"""
        return f"{os.path.splitext(filename)[0]}_chunk_{chunk_index:04d}"

    def is_document_processed(self, filename: str) -> bool:
        """Check if a document has already been processed"""
        try:
            with self.connection_pool.acquire() as connection:
                cursor = connection.cursor()
                
                # Check if any chunks from this file exist
                base_name = os.path.splitext(filename)[0]
                title = base_name.replace('_', ' ').title()
                
                cursor.execute(f"""
                    SELECT COUNT(*) FROM {self.table_name} 
                    WHERE TITLE = :1 OR DOCID LIKE :2
                """, [title, f"{base_name}_chunk_%"])
                
                count = cursor.fetchone()[0]
                return count > 0
                
        except Exception as e:
            self.logger.error(f"Failed to check if document is processed: {e}")
            return False

    def get_processed_files(self) -> List[str]:
        """Get list of already processed files"""
        try:
            with self.connection_pool.acquire() as connection:
                cursor = connection.cursor()
                
                cursor.execute(f"""
                    SELECT DISTINCT TITLE FROM {self.table_name}
                    WHERE TITLE IS NOT NULL
                """)
                
                results = cursor.fetchall()
                return [row[0] for row in results]
                
        except Exception as e:
            self.logger.error(f"Failed to get processed files: {e}")
            return []

    def store_documents_batch(self, documents_data):
        """Store multiple documents in a single transaction for better performance"""
        try:
            with self.connection_pool.acquire() as connection:
                cursor = connection.cursor()
                
                insert_sql = f"""
                INSERT INTO {self.table_name} 
                (DOCID, BODY, VECTOR, CHUNKID, URL, TITLE, PAGE_NUMBERS)
                VALUES (:1, :2, :3, :4, :5, :6, :7)
                """
                
                # Prepare batch data
                batch_data = []
                for doc_data in documents_data:
                    vector_array = array.array('f', doc_data['vector'])
                    batch_data.append([
                        doc_data['docid'],
                        doc_data['body'],
                        vector_array,
                        doc_data['chunkid'],
                        doc_data['url'],
                        doc_data['title'],
                        doc_data['page_numbers']
                    ])
                
                cursor.executemany(insert_sql, batch_data)
                connection.commit()
                
                self.logger.info(f"Stored batch of {len(documents_data)} documents in Oracle")
                
        except Exception as e:
            self.logger.error(f"Failed to store document batch: {e}")
            raise

    def store_document_in_oracle(self, docid: str, body: str, vector: List[float], 
                               chunkid: str = None, url: str = None, 
                               title: str = None, page_numbers: str = None):
        """Store document and its embedding in Oracle 23ai"""
        try:
            with self.connection_pool.acquire() as connection:
                cursor = connection.cursor()
                
                # Convert vector to array.array for Oracle VECTOR type
                if isinstance(vector, list):
                    vector_array = array.array('f', vector)  # 'f' for float32
                elif isinstance(vector, np.ndarray):
                    vector_array = array.array('f', vector.tolist())
                else:
                    vector_array = array.array('f', vector)
                
                insert_sql = f"""
                INSERT INTO {self.table_name} 
                (DOCID, BODY, VECTOR, CHUNKID, URL, TITLE, PAGE_NUMBERS)
                VALUES (:1, :2, :3, :4, :5, :6, :7)
                """
                
                cursor.execute(insert_sql, [
                    docid,
                    body,
                    vector_array,
                    chunkid,
                    url,
                    title,
                    page_numbers
                ])
                
                connection.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store document {docid}: {e}")
            raise


    def similarity_search(self, query_text: str, top_k: int = 5) -> List[dict]:
        """Perform similarity search using Oracle's vector capabilities"""
        try:
            query_embedding = self.embed_text(query_text)
            
            # Convert to array.array for Oracle VECTOR type
            if isinstance(query_embedding, list):
                query_array = array.array('f', query_embedding)
            elif isinstance(query_embedding, np.ndarray):
                query_array = array.array('f', query_embedding.tolist())
            else:
                query_array = array.array('f', query_embedding)
            
            with self.connection_pool.acquire() as connection:
                cursor = connection.cursor()
                
                # Use WITH clause to calculate distance once
                search_sql = f"""
                WITH ranked_docs AS (
                    SELECT DOCID, BODY, CHUNKID, URL, TITLE, PAGE_NUMBERS,
                        VECTOR_DISTANCE(VECTOR, :1, DOT) as distance
                    FROM {self.table_name}
                )
                SELECT DOCID, BODY, CHUNKID, URL, TITLE, PAGE_NUMBERS, distance
                FROM ranked_docs
                ORDER BY distance
                FETCH FIRST :2 ROWS ONLY
                """
                
                cursor.execute(search_sql, [query_array, top_k])
                results = cursor.fetchall()
                
                formatted_results = []
                for row in results:
                    # Convert CLOB to string if needed
                    body_text = row[1]
                    if hasattr(body_text, 'read'):  # It's a CLOB
                        body_text = body_text.read()
                    elif body_text is None:
                        body_text = ""
                    else:
                        body_text = str(body_text)
                    
                    formatted_results.append({
                        'docid': row[0],
                        'body': body_text,
                        'chunkid': row[2],
                        'url': row[3],
                        'title': row[4],
                        'page_numbers': row[5],
                        'distance': float(row[6])
                    })
                
                return formatted_results
                
        except Exception as e:
            self.logger.error(f"Failed to perform similarity search: {e}")
            raise



    def load_and_split_documents(self, file_path: str) -> List[Document]:
        """Process EPUB files and store them in Oracle with batch processing (idempotent)"""
        documents = []
        
        # Get list of already processed files
        processed_files = self.get_processed_files()
        self.logger.info(f"Already processed files: {len(processed_files)}")
        for pf in processed_files:
            self.logger.info(f"  - {pf}")
        
        epub_files = [f for f in os.listdir(file_path) if f.endswith(".epub")]
        self.logger.info(f"Found {len(epub_files)} EPUB files to process")
        
        for filename in epub_files:
            # Check if already processed (idempotency)
            if self.is_document_processed(filename):
                title = os.path.splitext(filename)[0].replace('_', ' ').title()
                self.logger.info(f"Skipping {filename} - already processed as '{title}'")
                continue
                
            epub_path = os.path.join(file_path, filename)
            print(f"Processing: {epub_path}")
            
            try:
                full_text = self.extract_text_from_epub(epub_path)
                title = os.path.splitext(filename)[0].replace('_', ' ').title()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                text_chunks = text_splitter.split_text(full_text)
                
                print(f"  - Split into {len(text_chunks)} chunks")
                print(f"  - Generating embeddings...")
                
                # Use batch embedding for better performance
                embeddings = self.embed_texts_batch(text_chunks, batch_size=16)
                
                print(f"  - Storing in Oracle...")
                
                # Prepare batch data for faster insertion
                batch_data = []
                for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
                    docid = self.generate_doc_id(filename, i)
                    chunkid = self.generate_chunk_id(filename, i)
                    
                    batch_data.append({
                        'docid': docid,
                        'body': chunk,
                        'vector': embedding,
                        'chunkid': chunkid,
                        'url': None,
                        'title': title,
                        'page_numbers': None
                    })
                    
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "docid": docid,
                            "chunkid": chunkid,
                            "filename": filename,
                            "title": title,
                            "chunk_index": i,
                            "total_chunks": len(text_chunks)
                        }
                    )
                    documents.append(doc)
                
                # Store in batches for better performance
                batch_size = 100
                for i in range(0, len(batch_data), batch_size):
                    batch = batch_data[i:i + batch_size]
                    self.store_documents_batch(batch)
                
                print(f"  - Completed: {filename}")
                
                # Clear GPU cache after each file
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                self.logger.error(f"Failed to process {filename}: {e}")
                continue
        
        return documents

    def create_vectorstore(self, documents: List[Document]) -> None:
        """Vector store creation is handled by Oracle storage"""
        self.logger.info(f"Documents stored in Oracle 23ai vector database: {len(documents)} chunks")
        return None

    def query_documents(self, query: str, top_k: int = 5) -> List[dict]:
        """Query documents using vector similarity search"""
        return self.similarity_search(query, top_k)

    def get_document_stats(self) -> dict:
        """Get statistics about stored documents"""
        try:
            with self.connection_pool.acquire() as connection:
                cursor = connection.cursor()
                
                cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                total_docs = cursor.fetchone()[0]
                
                cursor.execute(f"SELECT COUNT(DISTINCT TITLE) FROM {self.table_name}")
                unique_titles = cursor.fetchone()[0]
                
                cursor.execute(f"SELECT COUNT(DISTINCT CHUNKID) FROM {self.table_name} WHERE CHUNKID IS NOT NULL")
                total_chunks = cursor.fetchone()[0]
                
                return {
                    'total_documents': total_docs,
                    'unique_titles': unique_titles,
                    'total_chunks': total_chunks
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get document stats: {e}")
            return {}

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'connection_pool'):
            self.connection_pool.close()
            self.logger.info("Oracle connection pool closed")

def main():
    # Load environment variables
    load_dotenv()
    
    print("Environment variables check:")
    print(f"ORACLE_USER: {os.getenv('ORACLE_USER', 'NOT SET')}")
    print(f"ORACLE_DSN: {os.getenv('ORACLE_DSN', 'NOT SET')}")
    print(f"ORACLE_PASSWORD: {'SET' if os.getenv('ORACLE_PASSWORD') else 'NOT SET'}")
    print()
    
    rag = RAGPipeline(model_name="mistral-nemo:12b", max_memory_gb=3.0)
    
    try:
        documents = rag.load_and_split_documents("./epubs")
        rag.create_vectorstore(documents)
        
        stats = rag.get_document_stats()
        print(f"\nDatabase Statistics:")
        print(f"Total documents: {stats.get('total_documents', 0)}")
        print(f"Unique titles: {stats.get('unique_titles', 0)}")
        print(f"Total chunks: {stats.get('total_chunks', 0)}")
        
        query = "What is machine learning?"
        results = rag.query_documents(query, top_k=3)
        
        print(f"\nQuery: {query}")
        print("Results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']} (Distance: {result['distance']:.4f})")
            print(f"   Doc ID: {result['docid']}")
            print(f"   Chunk ID: {result['chunkid']}")
            
            # Safely handle the body text (which might be a CLOB)
            body_text = result['body']
            if body_text:
                # Truncate for display
                display_text = body_text[:200] + "..." if len(body_text) > 200 else body_text
                print(f"   Content: {display_text}")
            else:
                print(f"   Content: [No content]")
            print()
            
    except Exception as e:
        logging.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rag.cleanup()


if __name__ == "__main__":
    main()
