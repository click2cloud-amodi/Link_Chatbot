# Web_Scraping_Chatbot (Sugarcane)

## Overview
This project implements a web scraping-based question-answering (QA) chatbot for retrieving sugarcane-related information from web sources. It uses BeautifulSoup for web scraping, FAISS for vector storage, Azure OpenAI for embeddings, and LaMini-T5 for text generation. The chatbot processes scraped content into vector embeddings, retrieves relevant information, and generates responses based on user queries.

Model: text-embeddings-3-large → Creates embeddings and retrieves relevant text.
FAISS (Facebook AI Similarity Search) → Stores and searches embeddings efficiently.
Model: LaMini-T5 (developed by UAE University) → Generates refined responses and maintains conversational coherence.

## Inputs
sugarcane_testing_link.txt: A text file containing URLs to scrape.

## Outputs
1. Answers to User Queries
Retrieves and presents relevant information from scraped sources.
Uses FAISS and Azure OpenAI embeddings for efficient text retrieval.
Answers user questions using LaMini-T5, ensuring coherence and relevance.
2. Source References
Displays the URLs of retrieved information for transparency.
If no relevant content is found, the chatbot notifies the user.
3. Embedded Context
Extracted text is processed into semantic embeddings.
Enables fast and accurate retrieval for queries.

## Features
Web Scraping: Extracts text content from specified URLs, filtering out ads and irrelevant elements.
Text Processing: Cleans, tokenizes, and lemmatizes extracted text for improved NLP performance.
FAISS Vector Storage: Converts text into embeddings and stores them for quick retrieval.
LaMini-T5 Model: Uses a fine-tuned Transformer model to generate responses.
Azure OpenAI Embeddings: Transforms text into numerical representations for efficient search.
Persistent FAISS Database: Saves and loads embeddings to avoid redundant processing.
Retry Mechanism: Handles request failures with automatic retries.
Interactive Chatbot: Engages users in a conversational format.

## How It Works
1. Scraping URLs
Reads URLs from sugarcane_testing_link.txt.
Fetches content from the web and extracts text from paragraphs, headings, and tables.
Removes advertisements and non-informative elements.
2. Creating Embeddings
Processes extracted text into semantic embeddings using text-embedding-3-large.
Stores embeddings in a FAISS vector database for efficient search.
3. Retrieval and Answer Generation
When a user asks a question, the chatbot:
Searches FAISS for the most relevant text chunks.
Uses LaMini-T5 to refine and generate a natural response.
Provides source references for transparency.

## Notes
1. Ensure the URLs provided are accessible and contain sufficient text content.
2. Processing large websites with many links may take significant time and resources.

