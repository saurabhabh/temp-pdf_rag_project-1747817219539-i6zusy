import pinecone
import os
import uuid
import numpy as np
from dotenv import load_dotenv
from logging_config import setup_logging

load_dotenv()
logger = setup_logging()

def initialize_pinecone():
    """Initialize Pinecone and create/connect to index."""
    logger.info("Initializing Pinecone client")
    try:
        pc = pinecone.Pinecone(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
        index_name = os.getenv("PINECONE_INDEX_NAME", "pdf-embeddings")
        logger.debug(f"Checking for index: {index_name}")
        if index_name not in pc.list_indexes().names():
            logger.info(f"Creating new index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=3072,
                metric="cosine",
                spec=pinecone.ServerlessSpec(
                    cloud="azure",
                    region=os.getenv("PINECONE_ENVIRONMENT")
                )
            )
        else:
            logger.info(f"Index {index_name} already exists")
        return pc.Index(index_name)
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {str(e)}")
        raise

def delete_index():
    """Delete the Pinecone index."""
    logger.info("Deleting Pinecone index: pdf-embeddings")
    try:
        pc = pinecone.Pinecone(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
        index_name = os.getenv("PINECONE_INDEX_NAME", "pdf-embeddings")
        if index_name in pc.list_indexes().names():
            pc.delete_index(index_name)
            logger.info(f"Successfully deleted index: {index_name}")
        else:
            logger.warning(f"Index {index_name} does not exist")
    except Exception as e:
        logger.error(f"Failed to delete index: {str(e)}")
        raise

def pad_embedding(embedding, target_dim=3072):
    """Pad embedding to match target dimension."""
    if len(embedding) < target_dim:
        return embedding + [0.0] * (target_dim - len(embedding))
    return embedding[:target_dim]

def store_in_pinecone(text_chunks, images_with_context, pdf_name, index, batch_size=50):
    """Store text and image context embeddings in Pinecone in batches."""
    logger.info(f"Preparing to store {len(text_chunks)} text chunks and {len(images_with_context)} images for {pdf_name}")
    vectors = []
    try:
        # Store text embeddings
        for i, chunk in enumerate(text_chunks):
            from embedding_generator import get_text_embeddings
            logger.debug(f"Generating text embedding for chunk {i+1}/{len(text_chunks)}")
            embedding = get_text_embeddings(chunk)
            if embedding:
                vector_id = f"{pdf_name}_text_{uuid.uuid4()}"
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {"type": "text", "text": chunk, "pdf_name": pdf_name}
                })
            
            if len(vectors) >= batch_size:
                logger.info(f"Upserting batch of {len(vectors)} vectors")
                index.upsert(vectors=vectors)
                logger.info(f"Successfully stored {len(vectors)} vectors")
                vectors = []
        
        # Store image context embeddings
        for i, (image_path, context) in enumerate(images_with_context):
            from embedding_generator import get_text_embeddings
            logger.debug(f"Generating context embedding for image {i+1}/{len(images_with_context)}: {image_path}")
            embedding = get_text_embeddings(context if context else "Image without context")
            if embedding:
                vector_id = f"{pdf_name}_image_{uuid.uuid4()}"
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {"type": "image", "image_path": image_path, "context": context, "pdf_name": pdf_name}
                })
            
            if len(vectors) >= batch_size:
                logger.info(f"Upserting batch of {len(vectors)} vectors")
                index.upsert(vectors=vectors)
                logger.info(f"Successfully stored {len(vectors)} vectors")
                vectors = []
        
        if vectors:
            logger.info(f"Upserting final batch of {len(vectors)} vectors")
            index.upsert(vectors=vectors)
            logger.info(f"Successfully stored {len(vectors)} vectors")
    except Exception as e:
        logger.error(f"Error storing vectors in Pinecone: {str(e)}")
        raise