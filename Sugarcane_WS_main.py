import os
import time
import re
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv  # Import dotenv to load .env file
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# LangChain and Vector Store Imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline

# PyTorch and Transformers Imports for LaMini-T5
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Azure OpenAI Embeddings Import
from langchain_openai import AzureOpenAIEmbeddings

# Set workaround for OpenMP duplicate runtime issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==============================
# Constants and Configuration
# ==============================
FAISS_DB_DIR = 'faiss_index_sample_testing_1'
PROCESSED_LINKS_FILE = 'processed_links_sample_testing_1.txt'
MAX_RETRIES = 3
RETRY_DELAY = 5

# LaMini-T5 Model Checkpoint
CHECKPOINT = "MBZUAI/LaMini-T5-738M"

# ==============================
# Load Environment Variables
# ==============================
load_dotenv()

# ==============================
# Text Cleaning Function
# ==============================
def clean_text(text: str) -> str:
    """
    Cleans the input text by:
      - Removing extra whitespace.
      - Removing non-printable characters.
      - Tokenizing and lemmatizing the text.
    """
    # Remove multiple spaces and newline characters
    text = re.sub(r'\s+', ' ', text)
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E]', '', text)
    # Strip leading and trailing spaces
    text = text.strip()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Rejoin tokens into a single string
    processed_text = ' '.join(lemmatized_tokens)
    
    return processed_text

# ==============================
# Initialize LaMini-T5 Model
# ==============================
def initialize_llm() -> HuggingFacePipeline:
    """
    Initializes and returns the LaMini-T5 model pipeline wrapped in a HuggingFacePipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        CHECKPOINT,
        device_map=torch.device('cpu'),
        torch_dtype=torch.float32
    )
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=1024,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
    )
    return HuggingFacePipeline(pipeline=pipe)

llm = initialize_llm()

# ==============================
# Initialize Azure OpenAI Embeddings
# ==============================
# Load API configuration from environment variables
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_model = os.getenv("AZURE_OPENAI_MODEL", "text-embedding-3-large")
openai_api_version = os.getenv("OPENAI_API_VERSION", "2023-05-15")

embeddings_client = AzureOpenAIEmbeddings(
    openai_api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    model=azure_model,
    openai_api_version=openai_api_version
)

# ==============================
# Link Processing Functions
# ==============================
def get_processed_links() -> set:
    """
    Reads the processed links from file and returns a set of URLs.
    """
    if os.path.exists(PROCESSED_LINKS_FILE):
        with open(PROCESSED_LINKS_FILE, 'r') as f:
            return set(line.strip() for line in f)
    return set()

def mark_link_as_processed(url: str) -> None:
    """
    Marks a URL as processed by appending it to the processed links file.
    """
    with open(PROCESSED_LINKS_FILE, 'a') as file:
        file.write(url + '\n')

# ==============================
# Web Scraping Function with Data Processing
# ==============================
def scrape_website(url: str) -> str:
    """
    Scrapes the website at the given URL, cleans the HTML by removing ads,
    extracts text from paragraphs, headings, and tables, cleans the text,
    and returns the processed text content. Additionally, prints the full scraped text.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove potential advertisement elements
            for ad in soup.find_all(['div', 'span'], class_=lambda x: x and 'ad' in x.lower()):
                ad.decompose()

            # Print the cleaned HTML (first 1000 characters)
            print(f"\nCleaned HTML for {url} (first 1000 chars):\n", soup.prettify()[:1000], "...\n")

            # Extract text from paragraphs and headings
            text = ' '.join(p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3']))

            # Extract text from tables
            for table in soup.find_all('table'):
                table_text = ' '.join(td.get_text() for td in table.find_all('td'))
                text += f" {table_text}"

            # Process the extracted text
            processed_text = clean_text(text)

            # Print the complete processed text to terminal
            print(f"Processed text from {url}:\n{processed_text}\n")
            return processed_text
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed for {url}: {e}")
            time.sleep(RETRY_DELAY)

    print(f"Failed to retrieve {url} after {MAX_RETRIES} attempts.")
    return ""

# ==============================
# FAISS and QA Chain Initialization
# ==============================
def initialize_qa_chain(unprocessed_links: list) -> RetrievalQA:
    """
    Initializes the FAISS vector database and the RetrievalQA chain.
    If an existing FAISS index is found, the user is prompted to load it.
    Otherwise, documents are scraped from unprocessed URLs, split, and embedded.
    """
    if os.path.exists(FAISS_DB_DIR):
        load_existing = input("Load existing FAISS vectorDB? (y/n): ").strip().lower()
        if load_existing == 'y':
            print("Loading existing FAISS vectorDB...")
            vectordb = FAISS.load_local(
                FAISS_DB_DIR,
                embeddings_client,
                allow_dangerous_deserialization=True
            )
            print("FAISS vectorDB loaded successfully.")
            return RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
                return_source_documents=True
            )

    print("Creating new embeddings from provided URLs...")
    documents = []
    processed_links = get_processed_links()

    while unprocessed_links:
        url = unprocessed_links.pop(0)
        if url in processed_links:
            print(f"Skipping already processed URL: {url}")
            continue

        print(f"Processing: {url}")
        website_text = scrape_website(url)
        if website_text:
            print(f"Scraped {len(website_text)} characters from {url}")
            documents.append(Document(page_content=website_text, metadata={"source": url}))
            mark_link_as_processed(url)

    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=90)
    splits = text_splitter.split_documents(documents)

    if not splits:
        raise ValueError("No documents were processed, unable to create vector database.")

    # Create and save FAISS vector database
    vectordb = FAISS.from_documents(splits, embeddings_client)
    vectordb.save_local(FAISS_DB_DIR)
    print(f"FAISS vectorDB saved at {FAISS_DB_DIR}.")

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )

# ==============================
# Answer Processing Function
# ==============================
def process_answer(instruction: str, qa_chain: RetrievalQA) -> str:
    """
    Processes a user's query using the QA chain, extracts the answer and source documents,
    and formats the response. Returns a default message if no answer is found.
    """
    result = qa_chain.invoke({"query": instruction})
    source_docs = result.get('source_documents', [])
    answer = result['result'].strip()

    # Check for vague or non-answers
    if not source_docs or any(phrase in answer.lower() for phrase in 
                              ["does not provide", "couldn't find", "i don't know", "not mentioned"]):
        return "Sorry, I couldn't find the answer to your question."

    sources = list({doc.metadata.get("source", "Unknown source") for doc in source_docs})
    source_links = "\n".join(sources) if sources else "Unknown source"

    return f"{answer.capitalize()}\n\nSources:\n{source_links}"

# ==============================
# Utility Function to Read URLs
# ==============================
def get_urls_from_file(filename: str) -> list:
    """
    Reads a list of URLs from a file, ignoring empty lines.
    """
    if not os.path.exists(filename):
        print(f"{filename} not found.")
        return []
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line.strip()]

# ==============================
# Main Function
# ==============================
def main():
    """
    Main execution flow:
      - Reads URLs from a file.
      - Initializes the QA chain with embeddings.
      - Processes user queries in a loop.
    """
    filename = 'sugarcane_testing_link.txt'
    urls = get_urls_from_file(filename)

    if not urls:
        print(f"No URLs found in {filename}. Exiting.")
        return

    print("Processing embeddings. This may take some time...")
    qa_chain = initialize_qa_chain(urls)
    print("Embeddings processed. You can now ask questions about Sugarcane.")

    while True:
        prompt = input("\nYou: ")
        if prompt.lower() in ["exit", "quit", "bye"]:
            print("Exiting the chatbot.")
            break
        response = process_answer(prompt, qa_chain)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()

