PDF RAG Project
This project extracts text and individual images (e.g., figures, charts) from a PDF, generates vector embeddings using OpenAI's text-embedding-3-large for text and image context, stores them in Pinecone, and queries the data using Retrieval-Augmented Generation (RAG). Queries return text answers and paths to relevant images based on their context (e.g., captions or nearby text).
Prerequisites

Python 3.8+
OpenAI API key (https://platform.openai.com/)
Pinecone API key (https://www.pinecone.io/)
A text-based PDF file with images (e.g., sample.pdf)

Setup Instructions

Clone the repository (if applicable) or create the project structure:
pdf_rag_project/
├── images/
├── logs/
│   └── app.log
├── src/
│   ├── __init__.py
│   ├── logging_config.py
│   ├── pdf_processor.py
│   ├── embedding_generator.py
│   ├── pinecone_manager.py
│   ├── rag_query.py
│   ├── main.py
│   ├── query_interface.py
│   └── delete_index.py
├── .env
├── requirements.txt
└── README.md


Create and activate a virtual environment:
python -m venv venv


Windows: venv\Scripts\activate
macOS/Linux: source venv/bin/activate


Install dependencies:
pip install -r requirements.txt


Configure environment variables:

Create a .env file in the project root.
Add your API keys:OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-west1-gcp




Add your PDF:

Place your PDF file (e.g., sample.pdf) in the project root.
Update pdf_path in src/main.py to match your PDF filename.


Populate the database:

Run the main script to extract text and images from the PDF and store embeddings in Pinecone:python src/main.py




Query the database:

Use the query interface to interactively query the Pinecone database:python src/query_interface.py


Enter queries at the prompt (e.g., "What does the chart on page 2 show?"), and type 'exit' to quit. Responses include text answers and paths to relevant images.


Troubleshooting:

If you encounter errors related to the Pinecone index (e.g., missing metadata), delete the index and repopulate:python src/delete_index.py
python src/main.py





Logging

Logs are written to logs/app.log and printed to the console.
Log levels: DEBUG (detailed), INFO (major steps), WARNING (non-critical issues), ERROR (failures).
Check logs/app.log for persistent records of execution details.

Usage

Main Script (main.py): Extracts text and individual images from the PDF, splits text into chunks, generates embeddings for text and image context, stores them in Pinecone, and runs a sample query ("What is the main topic of the document?").
Query Interface (query_interface.py): Allows interactive querying of the Pinecone database via the command line, returning text answers and image paths based on context.
Delete Index (delete_index.py): Deletes the Pinecone index to resolve metadata issues.
To customize queries in main.py, modify the query variable. For custom queries, use query_interface.py or import query_rag from rag_query.py.

Notes

Ensure the PDF is text-based and contains images (e.g., figures, charts). For scanned PDFs, use an OCR library like pytesseract.
Images are saved in the images/ directory and retrieved based on their context (e.g., captions or nearby text).
Adjust chunk_size in pdf_processor.py, batch_size in pinecone_manager.py, or top_k in rag_query.py as needed.
Check Pinecone's free tier limits at https://www.pinecone.io/.
Image context is embedded using OpenAI's text embedding model for context-sensitive retrieval.

