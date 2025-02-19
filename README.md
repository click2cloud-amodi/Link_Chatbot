# Web_Scraping_Chatbot (Sugarcane)

## Overview
This project demonstrates how to create a question-answering (QA) chatbot using FAISS for vector storage and LangChain for building the retrieval-based QA chain. The chatbot scrapes content from provided URLs, processes the text into embeddings, and answers questions using a fine-tuned LaMini-T5 model.

Model = all-mpnet-base-v2 => to create embeddings and also to retrieve them
FAISS (Facebook AI Similarity Search) => VectorDB (to store the embeddings)
Model= Lamini-T5( developed by UAE University)=> to refine the response and to store the conversation

## Inputs
Sugarcane_links.txt`: File containing URLs to scrape

## Outputs
Answers to User Queries:
The chatbot provides contextually relevant answers to questions based on text scraped from the provided URLs.
Uses retrieval-based methods to find the most relevant text chunks and generate responses using a fine-tuned language model (LaMini-T5).

Source References:
For each answer, the chatbot includes the source URL from which the information was retrieved.
If no relevant information is found, it apologizes and notifies the user.

Embedded Context:
The text scraped from URLs is processed into semantic embeddings, allowing fast and accurate retrieval for queries.

## Features
1. Scrapes text content from URLs and internal links.
2. Creates and stores embeddings using FAISS vector database.
3. Leverages a fine-tuned language model (LaMini-T5) for answering questions.
4. Allows saving and loading FAISS vector databases for efficiency.
5. Maintains a log of successfully scraped links to avoid redundant processing.

Supports retry mechanism for failed requests.

Interactive chatbot interface for user queries.

The program will ask whether to load an existing FAISS database or create a new one from scratch.
After processing embeddings, you can ask questions based on the content of the scraped URLs.

## How It Works
1. **Scraping URLs**: The script fetches content from the provided URLs and internal links, extracting text from paragraphs and headings.
2. **Creating Embeddings**: The text is split into chunks and processed into embeddings using the `all-mpnet-base-v2` model.
3. **QA Chain**: A retrieval-based QA chain is built using the LaMini-T5 model for text generation and FAISS for vector retrieval.
4. **Query Processing**: The chatbot retrieves relevant text chunks and generates responses based on user queries.

## Configuration
- `FAISS_DB_DIR`: Directory to save or load the FAISS vector database.
- `Sugarcane_links.txt`: File containing URLs to scrape.
- `MAX_RETRIES` and `RETRY_DELAY`: Configure retry attempts for failed network requests.
  
## Notes
1. Ensure the URLs provided are accessible and contain sufficient text content.
2. Processing large websites with many links may take significant time and resources.
