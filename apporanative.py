import streamlit as st
from typing import List
import re
from langchain_core.prompts import ChatPromptTemplate
import atexit

# Import after fixing the circular import in embedder_oracle.py
try:
    from oracle_embeddings_native import RAGPipeline
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

class StreamlitRAGChat:
    def __init__(self):
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Initialize RAG Pipeline - let it handle its own LLM and Oracle embeddings
        self.rag = self.get_or_create_rag_pipeline()
        
        # Set up prompt template
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Answer the question based on the provided context.
        
        Context:
        {context}
        
        Question: {question}
        
        Instructions:
        - Use the context information to answer the question
        - If the context doesn't contain relevant information, say so clearly
        - Be helpful and specific
        - You can use <think> tags to show your reasoning
        
        Answer: """)

    @st.cache_resource
    def get_or_create_rag_pipeline(_self):
        """Create and cache the RAG pipeline with Oracle embeddings"""
        try:
            # RAGPipeline now uses Oracle embeddings internally
            pipeline = RAGPipeline(
                model_name="ALL_MINILM_L12_V2",  # Oracle database model
                max_memory_gb=3.0
            )
            return pipeline
        except Exception as e:
            st.error(f"Failed to initialize RAG Pipeline: {e}")
            raise

    def ensure_connection(self):
        """Test the database connection"""
        try:
            stats = self.rag.get_document_stats()
            return True
        except Exception as e:
            st.error(f"Connection lost: {e}")
            return False

    def test_embeddings(self):
        """Test Oracle embeddings functionality with detailed info"""
        try:
            test_text = "This is a test embedding"
            embedding = self.rag.embed_text(test_text)
            
            if embedding and len(embedding) > 0:
                # Store the dimension for later use
                self.rag.embedding_dimension = len(embedding)
                return True
            else:
                return False
        except Exception as e:
            st.error(f"Embedding test failed: {e}")
            return False

    def get_embedding_info(self):
        """Get information about the current embedding setup"""
        try:
            # Get embedding dimension from the RAG pipeline
            dimension = getattr(self.rag, 'embedding_dimension', None)
            
            # If dimension not directly available, try to get it from a test embedding
            if dimension is None:
                try:
                    test_embedding = self.rag.embed_text("test")
                    dimension = len(test_embedding) if test_embedding else 'Unknown'
                    # Cache it for future use
                    if dimension != 'Unknown':
                        self.rag.embedding_dimension = dimension
                except:
                    dimension = 'Unknown'
            
            # Get model name from RAG pipeline
            model_name = getattr(self.rag, 'model_name', 'ALL_MINILM_L12_V2')
            
            # Since we're using Oracle embeddings, we know the provider
            provider = "Oracle Database 23ai"
            
            return {
                'provider': provider,
                'model': model_name,
                'dimension': dimension
            }
        except Exception as e:
            # Fallback to known values since we're using Oracle
            return {
                'provider': 'Oracle Database 23ai',
                'model': 'ALL_MINILM_L12_V2',
                'dimension': 384,  # Known dimension for ALL_MINILM_L12_V2
                'note': f'Using defaults (detection failed: {str(e)})'
            }

    def get_detailed_embedding_test(self):
        """Get detailed embedding test results"""
        try:
            test_texts = [
                "Hello world",
                "Machine learning",
                "Database query"
            ]
            
            results = []
            for text in test_texts:
                embedding = self.rag.embed_text(text)
                results.append({
                    'text': text,
                    'dimension': len(embedding) if embedding else 0,
                    'sample': embedding[:5] if embedding and len(embedding) >= 5 else embedding
                })
            
            return results
        except Exception as e:
            return [{'error': str(e)}]

    def split_into_segments(self, text):
        """Split text into normal and think segments"""
        segments = []
        current_pos = 0
        
        pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        for match in pattern.finditer(text):
            if current_pos < match.start():
                normal_text = text[current_pos:match.start()].strip()
                if normal_text:
                    segments.append(("normal", normal_text))
            
            think_content = match.group(1).strip()
            if think_content:
                segments.append(("think", think_content))
            
            current_pos = match.end()
        
        remaining_text = text[current_pos:].strip()
        if remaining_text:
            segments.append(("normal", remaining_text))
        
        return segments

    def display_formatted_message(self, text):
        """Display message with think tag formatting"""
        segments = self.split_into_segments(text)
        
        if not segments:
            st.write(text)
            return
        
        for seg_type, content in segments:
            if seg_type == "think":
                st.markdown(
                    f'<div style="color: #808080; font-style: italic; padding: 5px; '
                    f'border-left: 3px solid #ccc; margin: 5px 0;">'
                    f'<em>💭 {content}</em></div>', 
                    unsafe_allow_html=True
                )
            else:
                if content.strip():
                    st.write(content)

    def get_context_from_results(self, results: List[dict]) -> str:
        """Convert search results to context string"""
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'Unknown')
            body = result.get('body', '').strip()
            distance = result.get('distance', 0)
            relevance = 1 - distance
            
            # Include documents with reasonable relevance
            if relevance > 0.2:
                context_parts.append(f"""
Document {i}: "{title}" (Relevance: {relevance:.2f})
Content: {body}
---""")
        
        return "\n".join(context_parts) if context_parts else "No sufficiently relevant documents found."

    def generate_response(self, question: str, context: str) -> str:
        """Generate response using the RAG pipeline's LLM"""
        try:
            formatted_prompt = self.prompt_template.format(
                context=context,
                question=question
            )
            
            # Use whatever LLM the RAG pipeline has configured
            response = self.rag.llm.invoke(formatted_prompt)
            return response
            
        except Exception as e:
            return f"Error generating response: {e}"

    def perform_search(self, query: str, top_k: int = 3):
        """Perform similarity search using Oracle embeddings"""
        try:
            if not self.ensure_connection():
                raise Exception("Database connection failed")
            
            # This now uses Oracle embeddings internally
            results = self.rag.similarity_search(query, top_k=top_k)
            return results
        except Exception as e:
            st.error(f"Search failed: {e}")
            return []

    def display_sources(self, results: List[dict]):
        """Display source information"""
        if not results:
            return
            
        with st.expander(f"📚 Sources ({len(results)} documents)", expanded=False):
            for i, result in enumerate(results, 1):
                title = result.get('title', 'Unknown Document')
                docid = result.get('docid', 'Unknown ID')
                distance = result.get('distance', 0)
                relevance = 1 - distance
                body = result.get('body', '')
                
                # Color code by relevance
                if relevance > 0.7:
                    color = "🟢"
                elif relevance > 0.5:
                    color = "🟡"
                else:
                    color = "🔴"
                
                st.markdown(f"{color} **Source {i}: {title}**")
                st.markdown(f"*Relevance: {relevance:.3f} | Doc ID: {docid}*")
                
                preview = body[:300] + "..." if len(body) > 300 else body
                st.markdown(f"```\n{preview}\n```")
                st.markdown("---")

    def run(self):
        st.set_page_config(
            page_title="Oracle RAG Chat",
            page_icon="🤖",
            layout="centered"
        )
        
        # Display logo if available
        try:
            st.image("oralogo.png", width=200)
        except:
            st.markdown("### 🤖 Oracle RAG Chat")
        
        st.title("Oracle Vector Database RAG Chatbot")
        st.markdown("Ask questions about your documents stored in Oracle 23ai Vector Database")
        
        # Check connection and embeddings
        connection_ok = self.ensure_connection()
        embeddings_ok = self.test_embeddings()
        
        if not connection_ok:
            st.error("❌ Database connection failed")
            st.stop()
        elif not embeddings_ok:
            st.error("❌ Oracle embeddings not working")
            st.stop()
        else:
            st.success("✅ Connected to Oracle Database with Oracle Embeddings")
        
        # Sidebar
        with st.sidebar:
            st.markdown("## About")
            st.markdown("This chatbot uses Oracle 23ai Vector Database with Oracle native embeddings for semantic search and RAG.")
            
            # Model info (dynamically get from RAG pipeline)
            st.markdown("## Model Info")
            try:
                llm_info = getattr(self.rag.llm, 'model', 'Unknown LLM')
                st.info(f"LLM: {llm_info}")
                
                # Get embedding info
                embed_info = self.get_embedding_info()
                
                st.info(f"🔗 Embedding Provider: {embed_info['provider']}")
                st.info(f"🤖 Embedding Model: {embed_info['model']}")
                st.info(f"📐 Vector Dimension: {embed_info['dimension']}")
                
                # Show note if there was a detection issue
                if 'note' in embed_info:
                    st.warning(f"ℹ️ {embed_info['note']}")
                    
            except Exception as e:
                st.warning(f"Could not load model info: {e}")
                # Show fallback info
                st.info("🔗 Embedding Provider: Oracle Database 23ai")
                st.info("🤖 Embedding Model: ALL_MINILM_L12_V2")
                st.info("📐 Vector Dimension: 384")
            
            # Database stats
            st.markdown("## Database Stats")
            try:
                stats = self.rag.get_document_stats()
                if stats:
                    st.metric("Total Documents", stats.get('total_documents', 0))
                    st.metric("Unique Titles", stats.get('unique_titles', 0))
                    st.metric("Total Chunks", stats.get('total_chunks', 0))
            except Exception as e:
                st.error(f"Error loading stats: {e}")
            
            # Settings
            st.markdown("## Settings")
            top_k = st.slider("Number of sources to retrieve", 1, 20, 10)
            
            # Test search
            st.markdown("## Test Search")
            test_query = st.text_input("Test query:")
            if st.button("Test Search") and test_query:
                with st.spinner("Testing Oracle embeddings..."):
                    test_results = self.perform_search(test_query, top_k=3)
                    if test_results:
                        st.success(f"Found {len(test_results)} results")
                        best_relevance = max([1-r.get('distance', 0) for r in test_results])
                        st.info(f"Best relevance: {best_relevance:.3f}")
                    else:
                        st.error("No results found")
            
            # System tests
            st.markdown("## System Tests")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Test Connection"):
                    if self.ensure_connection():
                        st.success("✅ Connection OK")
                    else:
                        st.error("❌ Connection Failed")
            
            with col2:
                if st.button("Test Embeddings"):
                    if self.test_embeddings():
                        st.success("✅ Embeddings OK")
                        # Show dimension if available
                        if hasattr(self.rag, 'embedding_dimension'):
                            st.info(f"Dimension: {self.rag.embedding_dimension}")
                    else:
                        st.error("❌ Embeddings Failed")
            
            # Detailed embedding test
            if st.button("Detailed Embedding Test"):
                with st.spinner("Testing embeddings..."):
                    test_results = self.get_detailed_embedding_test()
                    
                    if test_results and 'error' not in test_results[0]:
                        st.success("✅ Detailed embedding test passed")
                        for result in test_results:
                            st.write(f"Text: '{result['text']}'")
                            st.write(f"Dimension: {result['dimension']}")
                            st.write(f"Sample values: {result['sample']}")
                            st.write("---")
                    else:
                        st.error("❌ Detailed embedding test failed")
                        if test_results and 'error' in test_results[0]:
                            st.error(test_results[0]['error'])
            
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
                if message["role"] == "assistant":
                    self.display_formatted_message(message["content"])
                    if "sources" in message:
                        self.display_sources(message["sources"])
                else:
                    st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👤"):
                st.write(prompt)
            
            # Generate assistant response
            with st.chat_message("assistant", avatar="🤖"):
                try:
                    # Search for relevant documents using Oracle embeddings
                    with st.spinner("Searching documents with Oracle embeddings..."):
                        search_results = self.perform_search(prompt, top_k=top_k)
                    
                    if not search_results:
                        response = "I couldn't find any relevant documents to answer your question. Please try rephrasing or check if the information exists in the database."
                        st.write(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        # Generate context and response
                        context = self.get_context_from_results(search_results)
                        
                        with st.spinner("Generating response..."):
                            response = self.generate_response(prompt, context)
                        
                        # Display response and sources
                        self.display_formatted_message(response)
                        self.display_sources(search_results)
                        
                        # Save to session state
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "sources": search_results
                        })
                        
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

def main():
    try:
        app = StreamlitRAGChat()
        app.run()
    except Exception as e:
        st.error(f"Failed to initialize application: {e}")
        st.markdown("Please check:")
        st.markdown("- Oracle database connection")
        st.markdown("- Environment variables in .env file")
        st.markdown("- Oracle embeddings model (ALL_MINILM_L12_V2) is loaded in database")
        st.markdown("- Ollama service is running")

if __name__ == "__main__":
    main()
