import logging
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

def create_retriever(documents, openai_api_key, k=5):
    if not documents:
        logger.warning("No documents provided to create_retriever. Returning None.")
        return None

    logger.info(f"Creating retriever for {len(documents)} documents...")
    
    logger.info("Initializing OpenAI embeddings...")
    embedding_function = OpenAIEmbeddings(model = "text-embedding-3-large",api_key=openai_api_key)

    logger.info("Creating Chroma vector store for dense retrieval...")
    vectorstore = Chroma.from_documents(documents, embedding_function)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    logger.info("Creating BM25 retriever for keyword retrieval...")
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 10

    logger.info("Creating ensemble retriever with weights [0.7 dense, 0.3 keyword]...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )
    
    logger.info("Retriever created successfully.")
    return ensemble_retriever