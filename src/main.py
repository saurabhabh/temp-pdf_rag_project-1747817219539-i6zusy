import os
from pdf_processor import extract_text_and_images, split_text
from pinecone_manager import initialize_pinecone, store_in_pinecone
from rag_query import query_rag
from logging_config import setup_logging

logger = setup_logging()

def main():
    pdf_path = "traffic.pdf"  # Replace with your PDF file path
    pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
    
    logger.info(f"Starting processing for PDF: {pdf_path}")
    try:
        # Initialize Pinecone
        logger.debug("Initializing Pinecone")
        index = initialize_pinecone()
        
        # Extract text and images
        logger.debug("Extracting text and images from PDF")
        text, images_with_context = extract_text_and_images(pdf_path)
        if text or images_with_context:
            if text:
                logger.debug("Splitting text into chunks")
                text_chunks = split_text(text)
            else:
                text_chunks = []
                logger.warning("No text extracted from PDF")
            
            logger.debug("Storing text chunks and images in Pinecone")
            store_in_pinecone(text_chunks, images_with_context, pdf_name, index)
            
            # Example query
            query = "What is the main topic of the document?"
            logger.debug(f"Executing sample query: {query}")
            response, image_paths = query_rag(query, index)
            logger.info(f"Query: {query}\nResponse: {response}")
            if image_paths:
                logger.info(f"Retrieved images: {', '.join(image_paths)}")
        else:
            logger.error("Failed to extract text or images from PDF")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()