from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
import logging
import psutil
import os
import chromadb


class RAGPipeline:
    def __init__(self, model_name: str = "llama2:7b-chat-q4", max_memory_gb: float = 3.0):
        self.setup_logging()
        self.check_system_memory(max_memory_gb)
        
        # Load the language model (LLM)
        self.llm = OllamaLLM(model="deepseek-r1:7b")
        
        # Initialize embeddings using a lightweight model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}  # Use CPU for efficiency
        )
        
        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_template("""
        Answer the question based only on the following context. Be concise.
        If you cannot find the answer in the context, say "I cannot answer this based on the provided context."
        
        Context: {context}
        Question: {question}
        Answer: """)

        # Initialize Chroma client - Fixed configuration for remote server
        self.chroma_client = chromadb.HttpClient(
            host="localhost",
            port=18000
        )
        # Initialize Chroma collection
        self.collection_name = "oreilly"
        self.collection = self.chroma_client.get_or_create_collection(self.collection_name)


    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def check_system_memory(self, max_memory_gb: float):
        available_memory = psutil.virtual_memory().available / (1024 ** 3)
        self.logger.info(f"Available system memory: {available_memory:.1f} GB")
        if available_memory < max_memory_gb:
            self.logger.warning("Memory is below recommended threshold.")

    def retriever(self, query: str):
        query_embedding = self.embeddings.embed_query(query)  # Convert query to 768D embedding

        results = self.collection.query(
            query_embeddings=[query_embedding],  # Use embedding instead of query text
            n_results=4 
        )
        
        return results['documents'][0]  # Returns list of retrieved documents


    def setup_rag_chain(self, collection):
        """
        Set up a RAG chain using ChromaDB collection.
        Args:
            collection: ChromaDB collection object
        Returns:
            A runnable chain for RAG
        """
        
        def format_docs(docs):
            return "\n\n".join(doc for doc in docs)
        
        # Create the RAG chain
        rag_chain = (
            {
                "context": lambda x: format_docs(self.retriever(x)),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain

    def query(self, chain, question: str) -> str:

        """
        Query the RAG chain with a question.
        Args:
            chain: The RAG chain
            question: Question to ask
        Returns:
            str: Response from the chain
        """
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        self.logger.info(f"Memory usage: {memory_usage:.1f} MB")
        
        try:
            return chain.invoke(question)
        except Exception as e:
            self.logger.error(f"Error querying chain: {str(e)}")
            raise




def main():
    rag = RAGPipeline(model_name="deepseek-r1:8b", max_memory_gb=3.0)

    chain = rag.setup_rag_chain(rag.collection)
    
    question = "What is Diffusion?"
    response = rag.query(chain, question)
    print(f"Question: {question}\nAnswer: {response}")

if __name__ == "__main__":
    main()   