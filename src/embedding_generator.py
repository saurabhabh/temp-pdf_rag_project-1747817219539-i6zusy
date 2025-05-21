from openai import OpenAI
import os
from dotenv import load_dotenv
from logging_config import setup_logging
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

load_dotenv()
logger = setup_logging()

# Initialize CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_text_embeddings(text):
    """Generate embeddings for text using OpenAI's text-embedding-3-large model."""
    logger.debug(f"Generating text embedding for text of length {len(text)}")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )
        embedding = response.data[0].embedding
        logger.info(f"Successfully generated text embedding with {len(embedding)} dimensions")
        return embedding
    except Exception as e:
        logger.error(f"Error generating text embedding: {str(e)}")
        return None

def get_image_embeddings(image_path):
    """Generate embeddings for an image using CLIP."""
    logger.debug(f"Generating image embedding for {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embedding = clip_model.get_image_features(**inputs).squeeze().numpy()
        logger.info(f"Successfully generated image embedding with {len(embedding)} dimensions")
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating image embedding for {image_path}: {str(e)}")
        return None