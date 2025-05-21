import fitz  # PyMuPDF
import os
from logging_config import setup_logging

logger = setup_logging()

def extract_text_and_images(pdf_path):
    """Extract text and individual images with context from a PDF file."""
    logger.info(f"Extracting text and images from PDF: {pdf_path}")
    text = ""
    images_with_context = []
    image_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')
    os.makedirs(image_dir, exist_ok=True)

    try:
        pdf_document = fitz.open(pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            # Extract text from the page
            page_text = page.get_text("text")
            if page_text:
                text += page_text + "\n"
            else:
                logger.warning(f"No text extracted from page {page_num + 1}")

            # Extract images from the page
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list, start=1):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_path = os.path.join(image_dir, f"{os.path.basename(pdf_path).replace('.pdf', '')}_page{page_num+1}_img{img_index}.{image_ext}")
                with open(image_path, "wb") as image_file:
                    image_file.write(image_bytes)
                logger.info(f"Saved image: {image_path}")

                # Extract nearby text as context
                image_rects = page.get_image_rects(xref)
                if image_rects:
                    image_rect = image_rects[0]
                    # Manually expand the rectangle by 100 units vertically
                    expanded_rect = fitz.Rect(
                        image_rect.x0, 
                        max(0, image_rect.y0 - 100),  # Expand upward, ensure y0 >= 0
                        image_rect.x1, 
                        image_rect.y1 + 100  # Expand downward
                    )
                    nearby_text = page.get_text("text", clip=expanded_rect)
                    if not nearby_text:
                        # Try a larger area if no text is found
                        expanded_rect = fitz.Rect(
                            image_rect.x0, 
                            max(0, image_rect.y0 - 150),
                            image_rect.x1, 
                            image_rect.y1 + 150
                        )
                        nearby_text = page.get_text("text", clip=expanded_rect)
                    images_with_context.append((image_path, nearby_text))
                    logger.debug(f"Extracted context for image {image_path}: {nearby_text[:50]}...")
                else:
                    logger.warning(f"No rectangle found for image {image_path}, using empty context")
                    images_with_context.append((image_path, ""))

        logger.info(f"Extracted {len(text)} characters and {len(images_with_context)} images from PDF")
        pdf_document.close()
        return text, images_with_context
    except Exception as e:
        logger.error(f"Error extracting text and images from {pdf_path}: {str(e)}")
        return None, []

def split_text(text, chunk_size=1000):
    """Split text into chunks for embedding."""
    logger.info(f"Splitting text of length {len(text)} into chunks")
    try:
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_length += len(word) + 1
            if current_length > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word) + 1
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text: {str(e)}")
        return []