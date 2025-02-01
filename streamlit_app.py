import streamlit as st
from typing import List
import re
from langchain_core.prompts import ChatPromptTemplate
from RAGPipeline import RAGPipeline

class StreamlitRAGChat:
    def __init__(self):
        st.set_page_config(
            page_title="RAG Chatbot",
            page_icon="🤖",
            layout="centered"
        )
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Initialize RAG Pipeline
        self.rag = RAGPipeline(model_name="deepseek-r1:8b", max_memory_gb=3.0)
        
        # Set up prompt template
        self.rag.prompt = ChatPromptTemplate.from_template("""
        Answer the question based only on the following context. Be concise.
        When analyzing information, enclose your analysis in <think> tags.
        If you cannot find the answer in the context, say "I cannot answer this based on the provided context."
        
        Context: {context}
        Question: {question}
        Answer: """)
        
        self.chain = self.rag.setup_rag_chain(self.rag.collection)

    def display_logo(self):
        try:
            st.image("logo.png", width=200)
        except Exception:
            st.warning("Logo not found")

    def split_into_segments(self, text):
        """Split text into alternating normal and think segments"""
        segments = []
        current_pos = 0
        
        # Find all think tag pairs
        pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        for match in pattern.finditer(text):
            # Add normal text before think tag
            if current_pos < match.start():
                segments.append(("normal", text[current_pos:match.start()]))
            
            # Add think text
            think_content = match.group(1)
            segments.append(("think", think_content))
            
            current_pos = match.end()
        
        # Add remaining normal text
        if current_pos < len(text):
            segments.append(("normal", text[current_pos:]))
        
        return segments

    def display_formatted_message(self, text):
        """Display message with proper formatting for each segment"""
        segments = self.split_into_segments(text)
        
        for seg_type, content in segments:
            if seg_type == "think":
                st.markdown(f'<span style="color: #808080; font-style: italic;">{content}</span>', unsafe_allow_html=True)
            else:
                if content.strip():  # Only display non-empty content
                    st.write(content)

    def run(self):
        self.display_logo()
        st.title("RAG Chatbot")
        
        # Sidebar
        with st.sidebar:
            st.markdown("## About")
            st.markdown("This is a RAG-powered chatbot that answers questions based on the provided context.")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
                if message["role"] == "assistant":
                    self.display_formatted_message(message["content"])
                else:
                    st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question..."):
            # Display user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👤"):
                st.write(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant", avatar="🤖"):
                try:
                    message_placeholder = st.empty()
                    message_placeholder.text("Thinking...")
                    response = self.rag.query(self.chain, prompt)
                    message_placeholder.empty()
                    self.display_formatted_message(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {str(e)}")

def main():
    chat_app = StreamlitRAGChat()
    chat_app.run()

if __name__ == "__main__":
    main()
