import base64
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.documents.elements import Title, Header, NarrativeText, ListItem, Text, Image, Table
from unstructured.partition.pdf import partition_pdf

# --- IMAGE AND TABLE SUMMARIZATION (Requires Vision Model) ---

def encode_image(image_path):
    """Encodes an image file into a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def summarize_image(encoded_image, openai_api_key):
    """Generates a summary for a base64-encoded image using GPT-4-vision."""
    prompt = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Describe the image in detail. Be specific about any text, data, or charts visible."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
            ]
        )
    ]
    chat = ChatOpenAI(model="gpt-4.1-2025-04-14", api_key=openai_api_key, max_tokens=2048)
    response = chat.invoke(prompt)
    return response.content

def summarize_table(table_html, openai_api_key):
    """Generates a summary for an HTML table using GPT-4."""
    prompt = f"Summarize the following table:\n\n{table_html}\n\nProvide a concise summary that captures the key information."
    chat = ChatOpenAI(model="gpt-4", api_key=openai_api_key, temperature=0)
    response = chat.invoke([HumanMessage(content=prompt)])
    return response.content

# --- CORE PDF PARTITIONING AND TEXT CHUNKING ---



def partition_and_chunk(pdf_path, use_enhanced_processing=False, openai_api_key=None, temp_dir="temp_data"):
    """
    Partitions a PDF and intelligently chunks the content based on titles and sections.
    This creates more contextually relevant chunks for better retrieval.
    """
    if use_enhanced_processing and not openai_api_key:
        raise ValueError("OpenAI API key is required for enhanced processing.")

    image_output_dir = os.path.join(temp_dir, "images")
    os.makedirs(image_output_dir, exist_ok=True)

    raw_pdf_elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res", 
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=False,
        extract_images_in_pdf=True, 
        extract_image_block_output_dir=image_output_dir,
        infer_table_structure=True,
    )

    documents = []
    current_title = ""
    current_text_block = ""

    for i, el in enumerate(raw_pdf_elements):
        metadata = {"source": os.path.basename(pdf_path), "page_number": el.metadata.page_number or 1, "element_id": i}

        if isinstance(el, (Title, Header)):
            # When a new title/header is found, save the previous text block as a document
            if current_text_block:
                documents.append(Document(page_content=f"{current_title}\n\n{current_text_block.strip()}", metadata=metadata))
            
            # Start a new block
            current_title = el.text
            current_text_block = ""

        elif isinstance(el, (NarrativeText, ListItem, Text)):
            current_text_block += el.text + "\n"
        
        elif use_enhanced_processing:
            # If we encounter an image or table, flush the current text block first
            if current_text_block:
                documents.append(Document(page_content=f"{current_title}\n\n{current_text_block.strip()}", metadata=metadata))
                current_text_block = ""
            
            # Then process the image or table
            if isinstance(el, Image):
                try:
                    image_path = el.metadata.image_path
                    encoded_image = encode_image(image_path)
                    summary = summarize_image(encoded_image, openai_api_key)
                    documents.append(Document(page_content=f"[Image under '{current_title}']\nSummary: {summary}", metadata=metadata))
                except Exception:
                    documents.append(Document(page_content=f"[Image under '{current_title}']", metadata=metadata))
            
            elif isinstance(el, Table) and el.metadata.text_as_html:
                try:
                    summary = summarize_table(el.metadata.text_as_html, openai_api_key)
                    documents.append(Document(page_content=f"[Table under '{current_title}']\nSummary: {summary}", metadata=metadata))
                except Exception:
                    documents.append(Document(page_content=f"[Table under '{current_title}']", metadata=metadata))

    # Add the last processed text block
    if current_text_block:
        documents.append(Document(page_content=f"{current_title}\n\n{current_text_block.strip()}", metadata=metadata))

    return documents