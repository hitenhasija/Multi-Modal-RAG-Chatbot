import logging
import os
from logging.handlers import RotatingFileHandler

# Define the log directory and file path
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

def setup_logger():
    """
    Sets up a rotating file logger for the application.
    - Creates a 'logs' directory if it doesn't exist.
    - Clears the log file on each new run for a clean view.
    - Formats logs to be clear and informative.
    """
    # Create the logs directory if it doesn't exist
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # Clear the log file at the start of a new session
    # This ensures the log viewer in Streamlit only shows logs for the current run
    if os.path.exists(LOG_FILE):
        open(LOG_FILE, 'w').close()
        
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Capture messages of level INFO and above

    # Prevent Streamlit from adding its own handlers, which can cause duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Create a handler that writes log messages to a file, with rotation
    # This prevents the log file from growing indefinitely
    handler = RotatingFileHandler(
        LOG_FILE, 
        maxBytes=1*1024*1024, # 1 MB per file
        backupCount=5        # Keep up to 5 old log files
    )

    # Create a log format that is easy to read
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    # Add the handler to the root logger
    logger.addHandler(handler)

    return logger

def get_log_file_path():
    """A helper function to get the path to the log file."""
    return LOG_FILE