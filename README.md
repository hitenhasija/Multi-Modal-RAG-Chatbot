# Advanced Multi-Modal RAG Chatbot

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://multi-modal-rag-chatbot-rtmmzvse2xr8cuznalvs3w.streamlit.app/)

This repository contains the source code for an advanced, conversational AI chatbot capable of analyzing and answering questions about complex PDF documents containing text, tables, and images.

---

## 🚀 Live Application Demo

The application is live and publicly accessible. You can test its full capabilities at the following link:

**[https://multi-modal-rag-chatbot-rtmmzvse2xr8cuznalvs3w.streamlit.app/](https://multi-modal-rag-chatbot-rtmmzvse2xr8cuznalvs3w.streamlit.app/)**


---

## ✨ Key Features

-   **🧠 Conversational Memory:** The chatbot can understand follow-up questions, pronouns, and context from the ongoing conversation, powered by a persistent Redis database backend.
-   **📄 Multi-Modal Data Processing:** Goes beyond simple text extraction by using `unstructured.io`'s sophisticated local inference models to parse complex PDF layouts.
-   **🖼️ Vision-Enabled Summarization:** When "Enhanced Processing" is activated, the app leverages OpenAI's `gpt-4-turbo` model to generate detailed, accurate summaries of diagrams, charts, and tables within the document.
-   **🔍 Hybrid Search for Maximum Relevance:** Employs an `EnsembleRetriever` that combines the strengths of dense vector search (for semantic meaning) and traditional keyword search (BM25, for specific terms), ensuring the most relevant context is always found.
-   **⚡ High-Speed, High-Quality Generation:** Utilizes the Groq API with the Llama 3 70B model to generate answers with extremely low latency.
-   **🛡️ Robust and Grounded:** A comprehensive system prompt ensures all answers are strictly based on the document's content, preventing hallucinations and providing safe, reliable responses.
-   **☁️ Fully Deployed:** The application is deployed on Streamlit Community Cloud, demonstrating a complete development-to-production workflow.

---

## 🛠️ Tech Stack & Architecture

-   **Frontend:** Streamlit
-   **Backend & Orchestration:** LangChain
-   **PDF Parsing:** `unstructured[local-inference]` with Tesseract for OCR
-   **AI Models:**
    -   **LLM for Generation:** Groq (llama-3.3-70b-versatile)
    -   **Vision & Summarization:** OpenAI (`gpt-4.1`)
    -   **Embeddings:** OpenAI (`text-embedding-3-large`)
-   **Databases:**
    -   **Vector Store:** ChromaDB
    -   **Chat History:** Redis
-   **Deployment:** Streamlit Community Cloud

---

## 🔧 Local Setup and Installation

To run this application on your local machine, follow these steps.

**1. System Dependencies:**
   - **Tesseract OCR:** Required by `unstructured`.
     - **Windows:** Download from the [official Tesseract page](https://github.com/UB-Mannheim/tesseract/wiki).
     - **macOS:** `brew install tesseract`
     - **Debian/Ubuntu:** `sudo apt-get install tesseract-ocr`
   - **Redis:** Required for chat history. The easiest way to run a local instance is with Docker.
     ```bash
     docker run -d --name my-redis-container -p 6379:6379 redis
     ```

**2. Clone and Set Up the Environment:**
   ```bash
   # Clone the repository
   git clone https://github.com/hitenhasija/Multi-Modal-RAG-Chatbot.git
   cd Multi-Modal-RAG-Chatbot

   # Create and activate a Python virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install all required Python packages
   pip install -r requirements.txt
