from langchain_community.embeddings.oracleai import OracleEmbeddings
from langchain_core.documents import Document
import os
import oracledb
from ebooklib import epub
from dotenv import load_dotenv


"""
# using ocigenai
embedder_params = {
    "provider": "ocigenai",
    "credential_name": "OCI_CRED",
    "url": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText",
    "model": "cohere.embed-english-light-v3.0",
}

# using huggingface
embedder_params = {
    "provider": "huggingface", 
    "credential_name": "HF_CRED", 
    "url": "https://api-inference.huggingface.co/pipeline/feature-extraction/", 
    "model": "sentence-transformers/all-MiniLM-L6-v2", 
    "wait_for_model": "true"
}
"""

def setup_oracle_connection():
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
        
        print("Connecting to Oracle using provided DSN")
        
        try:
            conn = oracledb.connect(user=user, password=password, dsn=dsn)
            # Test with a simple query
            cursor = conn.cursor()
            cursor.execute("SELECT 'Connection successful' FROM DUAL")
            result = cursor.fetchone()
            print(f"Connection test result: {result[0]}")
            cursor.close()
            print("Successfully connected to Oracle database")
        except Exception as e:
            print(f"Failed to connect to Oracle: {e}")
            print("Please ensure your .env file contains:")
            print("- ORACLE_USER (your database username)")
            print("- ORACLE_PASSWORD (your database password)")
            print("- ORACLE_DSN (your full DSN connection string)")
            raise
      
        return conn         
    except Exception as e:
        print(f"Failed to connect to Oracle: {e}")
        print("Please ensure your .env file contains:")
        print("- ORACLE_USER (your database username)")
        print("- ORACLE_PASSWORD (your database password)")
        print("- ORACLE_DSN (your full DSN connection string)")
        raise

# Load environment variables
load_dotenv()

# using ONNX model loaded to Oracle Database
embedder_params = {"provider": "database", "model": "ALL_MINILM_L12_V2"}

# Get the actual connection object by calling the function
conn = setup_oracle_connection()

# If a proxy is not required for your environment, you can omit the 'proxy' parameter below
embedder = OracleEmbeddings(conn=conn, params=embedder_params)
embed = embedder.embed_query("Hello World!")
embed = embedder.embed_documents("Hello World!")


# Close the connection when done
conn.close()
print("Connection is closed.")

""" verify """
print(f"Embedding generated by OracleEmbeddings: {embed}")
