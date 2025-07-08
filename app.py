import streamlit as st
import os
import tempfile
import logging
import torch

# --- Import our modules, including the NEW chat history manager ---
from src.app_logging import setup_logger, get_log_file_path
from src.config import get_openai_api_key, get_groq_api_key, load_hf_token
from src.data_processing import partition_and_chunk
from src.vector_store import create_retriever
from src.chat_logic import get_rag_chain
from src.chat_history import get_message_history  # <-- NEW IMPORT

# --- 1. Setup Logger and GPU at the very beginning (NO CHANGE) ---
logger = setup_logger()
load_hf_token()

try:
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        logger.info("CUDA is available. Set default device to GPU 0.")
        st.toast("✅ NVIDIA GPU found and selected.", icon="🚀")
    else:
        logger.info("CUDA not available. Operations will run on CPU.")
        st.toast("⚠️ NVIDIA GPU not found. Using CPU.", icon="cpu")
except Exception as e:
    logger.error(f"Error during GPU selection: {e}")
    st.toast(f"GPU selection error: {e}", icon="🔥")

# --- Streamlit Page Configuration (NO CHANGE) ---
st.set_page_config(page_title="Chat with your PDF", layout="wide")
st.title("📄 Chat with Your Multi-Modal PDF")

# --- Session State Initialization ---
# We no longer store the 'messages' list here.
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# --- Helper Functions ---
def process_pdf(uploaded_file, use_enhanced):
    """Handles the processing of the uploaded PDF file and displays logs."""
    try:
        openai_key = get_openai_api_key()
        
        logger.info(f"Starting new processing job for file: {uploaded_file.name}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        log_container = st.expander("Processing Logs", expanded=True)
        log_placeholder = log_container.empty()
        log_placeholder.info("Starting processing...")
        
        with st.spinner("Processing PDF... See logs below for details."):
            # This entire processing block is unchanged
            documents = partition_and_chunk(
                pdf_path=tmp_file_path,
                use_enhanced_processing=use_enhanced,
                openai_api_key=openai_key,
                temp_dir=tempfile.gettempdir()
            )
            
            with open(get_log_file_path(), "r") as f:
                log_placeholder.code(f.read(), language="log")

            if not documents:
                logger.error("Partitioning returned no documents. Halting process.")
                st.error("Could not extract any content from the PDF. Please check the logs.")
                return

            logger.info("Creating document retriever...")
            retriever = create_retriever(documents, openai_key)
            
            with open(get_log_file_path(), "r") as f:
                log_placeholder.code(f.read(), language="log")

            if retriever is None:
                logger.error("Failed to create document retriever.")
                st.error("Failed to create the document retriever. Check logs for details.")
                return

            logger.info("Creating RAG chain with the LLM...")
            st.session_state.rag_chain = get_rag_chain(retriever)
            st.session_state.processed_file = uploaded_file.name
            
            # --- MODIFICATION: Reset chat session for the new document ---
            # This replaces the old `st.session_state.messages = []`
            if "session_id" in st.session_state:
                del st.session_state["session_id"]
            logger.info("✅ Chat session reset for new document.")
            # --- END OF MODIFICATION ---
            
            st.success("Your document has been processed! You can now ask questions.")
            
            with open(get_log_file_path(), "r") as f:
                log_placeholder.code(f.read(), language="log")

    except Exception as e:
        logger.error(f"A critical error occurred in the processing pipeline: {e}", exc_info=True)
        st.error(f"An error occurred during processing. Please check the logs for details.")
        if 'log_placeholder' in locals():
            try:
                with open(get_log_file_path(), "r") as f:
                    log_placeholder.code(f.read(), language="log")
            except FileNotFoundError:
                log_placeholder.error("Log file not found.")
    finally:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

# --- Sidebar (NO CHANGE) ---
with st.sidebar:
    st.header("1. Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    st.header("2. Configure Processing")
    use_enhanced_processing = st.checkbox(
        "Enable Enhanced Data Processing",
        help="Summarizes images and tables using OpenAI's models. Requires a funded OpenAI API key."
    )
    if st.button("Process Document"):
        if uploaded_file:
            process_pdf(uploaded_file, use_enhanced_processing)
        else:
            st.warning("Please upload a PDF file first.")

# --- Main Chat Interface ---
# --- Main Chat Interface ---
st.header("Ask Questions About Your Document")

if not st.session_state.rag_chain:
    st.info("Please upload and process a PDF using the sidebar to begin.")
else:
    # This is the new, correct logic for the conversational chain
    
    # Get the message history object connected to Redis
    history = get_message_history()
    
    # Display all past messages from the Redis history
    for message in history.messages:
        with st.chat_message(message.type):
            st.markdown(message.content)

    # Accept new user input
    if prompt := st.chat_input("What is your question?"):
        # Add user's message to Redis and display it
        history.add_user_message(prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display the assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # --- THE FIX ---
                    # 1. Call invoke with a dictionary containing the input and chat history
                    response_dict = st.session_state.rag_chain.invoke(
                        {"input": prompt, "chat_history": history.messages}
                    )
                    # 2. Extract the actual answer from the 'answer' key of the response
                    response = response_dict["answer"]
                    # --- END OF FIX ---
                    
                    st.markdown(response)
                    # Add the AI's response to Redis
                    history.add_ai_message(response)
                except Exception as e:
                    logger.error(f"Error during RAG chain invocation: {e}", exc_info=True)
                    st.error(f"An error occurred while getting the answer: {e}")