from pinecone_manager import delete_index
from logging_config import setup_logging

logger = setup_logging()

if __name__ == "__main__":
    try:
        delete_index()
    except Exception as e:
        logger.error(f"Error deleting index: {str(e)}")