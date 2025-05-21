from openai import OpenAI
import os
from dotenv import load_dotenv
from embedding_generator import get_text_embeddings
from logging_config import setup_logging

load_dotenv()
logger = setup_logging()

def query_rag(query, index, top_k=5):
    """Query Pinecone with RAG using OpenAI, returning text and image results."""
    logger.info(f"Processing query: {query}")
    try:
        logger.debug("Generating query embedding")
        query_embedding = get_text_embeddings(query)
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return "Error generating query embedding.", []
        
        logger.debug(f"Querying Pinecone with top_k={top_k}")
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        logger.info(f"Retrieved {len(results['matches'])} matches from Pinecone")
        
        text_context = ""
        image_data = []
        for match in results["matches"]:
            metadata = match.get("metadata", {})
            match_type = metadata.get("type")
            if match_type == "text":
                text_context += metadata.get("text", "") + "\n"
            elif match_type == "image":
                image_path = metadata.get("image_path")
                context = metadata.get("context", "")
                if image_path:
                    image_data.append((image_path, context))
            else:
                logger.warning(f"Match with ID {match['id']} has invalid or missing 'type' in metadata")
        
        logger.debug(f"Text context length: {len(text_context)} characters, {len(image_data)} images retrieved")
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.debug("Generating response with OpenAI")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the query. If images are relevant, mention them but do not describe their content."},
                {"role": "user", "content": f"Context:\n{text_context}\n\nQuery: {query}"}
            ]
        )
        answer = response.choices[0].message.content
        if image_data:
            answer += f"\n\nRelevant images found (see paths below)."
        logger.info("Successfully generated response")
        return answer, image_data
    except Exception as e:
        logger.error(f"Error during RAG query: {str(e)}")
        return f"Error generating response: {str(e)}", []