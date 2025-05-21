import pinecone
import os
import platform
import subprocess
from dotenv import load_dotenv
from rag_query import query_rag
from logging_config import setup_logging

load_dotenv()
logger = setup_logging()

def initialize_pinecone():
    """Initialize Pinecone and connect to the existing index."""
    logger.info("Initializing Pinecone client for query interface")
    try:
        pc = pinecone.Pinecone(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
        index_name = "pdf-embeddings"
        if index_name not in pc.list_indexes().names():
            logger.error(f"Index {index_name} does not exist. Run main.py to create and populate it.")
            raise ValueError(f"Index {index_name} does not exist. Please run main.py first.")
        logger.info(f"Connected to index: {index_name}")
        return pc.Index(index_name)
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {str(e)}")
        raise

def open_image(image_path):
    """Open an image file using the default system viewer."""
    try:
        if not os.path.exists(image_path):
            logger.warning(f"Image file not found: {image_path}")
            return False
        if platform.system() == "Windows":
            os.startfile(image_path)
        else:
            opener = "open" if platform.system() == "Darwin" else "xdg-open"
            subprocess.run([opener, image_path], check=True)
        logger.info(f"Opened image: {image_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to open image {image_path}: {str(e)}")
        return False

def query_interface():
    """Command-line interface for querying the Pinecone database."""
    logger.info("Starting query interface")
    try:
        index = initialize_pinecone()
        print("Welcome to the PDF RAG Query Interface!")
        print("Enter your query below. Type 'exit' to quit.")
        print("Images relevant to the query will be listed, and you can choose to open them.")

        while True:
            query = input("\nEnter your query: ").strip()
            if query.lower() == 'exit':
                logger.info("User exited the query interface")
                print("Exiting query interface.")
                break
            
            if not query:
                logger.warning("Empty query entered")
                print("Please enter a non-empty query.")
                continue
            
            logger.info(f"Received user query: {query}")
            response, image_data = query_rag(query, index)
            print(f"\nResponse: {response}")
            
            if image_data:
                print("\nRelevant images:")
                for img_path, context in image_data:
                    print(f"- {img_path}")
                    if context:
                        print(f"  Context: {context[:100]}{'...' if len(context) > 100 else ''}")
                    else:
                        print("  Context: No context available")
                # Prompt to open images
                open_images = input("\nWould you like to open these images? (y/n): ").strip().lower()
                if open_images == 'y':
                    for img_path, _ in image_data:
                        if open_image(img_path):
                            print(f"Opened: {img_path}")
                        else:
                            print(f"Failed to open: {img_path}")
            else:
                print("\nNo relevant images found.")
            logger.info("Query processed successfully")
    
    except Exception as e:
        logger.error(f"Error in query interface: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    query_interface()