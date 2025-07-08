# Multi-Modal RAG Chatbot for PDFs

This application allows you to chat with a PDF document. It uses a sophisticated RAG pipeline that processes text, tables, and images from the PDF to provide comprehensive answers.

## Features

- **Multi-Modal Processing:** Extracts and understands text, tables, and images from your PDF.
- **Hybrid Search:** Combines semantic (vector) search with keyword (BM25) search for robust document retrieval.
- **Optional Enhanced Processing:** You can enable summarization of images and tables using OpenAI's vision models if you have a funded API key.
- **Fast Generation:** Uses the high-speed Groq API for generating answers.
- **Chat Interface:** An easy-to-use chat interface built with Streamlit that remembers your conversation history.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd MultiModal-RAG-Chatbot
    ```

2.  **Install System Dependencies (for `unstructured`):**
    You may need to install Tesseract OCR.
    - On Debian/Ubuntu: `sudo apt-get install tesseract-ocr`
    - On macOS: `brew install tesseract`
    - On Windows: Download from the [official Tesseract page](https://github.com/UB-Mannheim/tesseract/wiki).

3.  **Create a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set up API Keys:**
    - Create a file named `.env` in the project's root directory.
    - Add your API keys to it:
      ```
      OPENAI_API_KEY="your_openai_api_key"
      GROQ_API_KEY="your_groq_api_key"
      ```
    - **Note:** The `OPENAI_API_KEY` is only required if you enable the "Enhanced Data Processing" option in the app. The core functionality works with just the `GROQ_API_KEY`.

## How to Run

1.  **Start the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
2.  Open your web browser and go to the local URL provided by Streamlit (usually `http://localhost:8501`).
3.  Upload a PDF file.
4.  (Optional) Check the "Enable Enhanced Data Processing" box if you have a funded OpenAI account to get summaries of images and tables.
5.  Wait for the processing to complete, then start asking questions!