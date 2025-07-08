import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

def get_openai_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        logger.error("OPENAI_API_KEY not found in environment variables.")
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    return key

def get_groq_api_key():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        logger.error("GROQ_API_KEY not found in environment variables.")
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    return key

def load_hf_token():
    token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        logger.info("Hugging Face token loaded from .env file.")
    else:
        logger.info("Hugging Face token not found in .env file. Proceeding without it.")
        

def get_redis_url():
    """Returns the Redis connection URL from environment variables."""
    url = os.getenv("REDIS_URL")
    if not url:
        logger.error("REDIS_URL not found in environment variables.")
        raise ValueError("REDIS_URL not found in environment variables.")
    return url