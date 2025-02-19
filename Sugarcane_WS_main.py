import os
import requests
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import Document
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
import time

# Define FAISS vector database directory
FAISS_DB_DIR = 'faiss_index_training'
SKIPPED_LINKS_FILE = 'skipped_links_training.txt'

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 5

# Function to check if a link is already scraped
def is_link_scraped(url):
    if os.path.exists(SKIPPED_LINKS_FILE):
        with open(SKIPPED_LINKS_FILE, 'r') as file:
            scraped_links = file.read().splitlines()
        return url in scraped_links
    return False

# Function to log successfully scraped links
def log_scraped_link(url):
    with open(SKIPPED_LINKS_FILE, 'a') as file:
        file.write(url + '\n')

# Function to scrape text content from a URL with retries
def scrape_website(url):
    if is_link_scraped(url):  
        print(f"Skipping already scraped link: {url}")
        return ""
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Check if request was successful
            soup = BeautifulSoup(response.content, 'html.parser')
            text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3'])])
            
            if text.strip():
                log_scraped_link(url)  # Save successful link
                return text
            else:
                print(f"No valid content found on {url}")
                return ""

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed to retrieve {url}: {e}")
            time.sleep(RETRY_DELAY)

    print(f"Failed to retrieve {url} after {MAX_RETRIES} attempts.")
    return ""

# Initialize the QA Chain
def initialize_qa_chain(urls):
    if os.path.exists(FAISS_DB_DIR) and input("Load existing FAISS vectorDB? (y/n): ").strip().lower() == 'y':
        print("Loading existing FAISS vectorDB...")
        vectordb = FAISS.load_local(FAISS_DB_DIR, SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2"), allow_dangerous_deserialization=True)
    else:
        print("Creating new embeddings from provided URLs...")
        documents = []

        # Scrape and load only new URLs (skip already scraped ones)
        for url in urls:
            website_text = scrape_website(url)
            if website_text:
                document = Document(page_content=website_text, metadata={"source": url})
                documents.append(document)

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=90)
        splits = text_splitter.split_documents(documents)

        # Create embeddings
        embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
        vectordb = FAISS.from_documents(splits, embeddings)

        # Save the vectorDB to the specified path
        vectordb.save_local(FAISS_DB_DIR)
        print(f"FAISS vectorDB saved at {FAISS_DB_DIR}.")

    # Initialize the LaMini-T5 model
    CHECKPOINT = "MBZUAI/LaMini-T5-738M"
    TOKENIZER = AutoTokenizer.from_pretrained(CHECKPOINT)
    BASE_MODEL = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT, device_map=torch.device('cpu'), torch_dtype=torch.float32)
    pipe = pipeline(
        'text2text-generation',
        model=BASE_MODEL,
        tokenizer=TOKENIZER,
        max_length=1024,
        do_sample=True,
        temperature=0.1,
        top_p=0.95,
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # Build a QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 10}),
        return_source_documents=True
    )

    return qa_chain

# Function to process the user's query
def process_answer(instruction, qa_chain):
    result = qa_chain.invoke({"query": instruction})
    source_docs = result.get('source_documents', [])
    answer = result['result'].strip()

    # Check if there are no source documents or if the answer is a fallback
    if len(source_docs) == 0 or "does not provide information" in answer.lower() or "couldn't find" in answer.lower() or "not mentioned" in answer.lower():
        return "Sorry, I couldn't find the answer to your question in the provided context."
    else:
        source_url = source_docs[0].metadata.get("source", "Unknown source")
        return f"{answer}\n\nSource: {source_url}"

# Function to read URLs from a file
def get_urls_from_file(filename):
    if not os.path.exists(filename):
        print(f"{filename} not found.")
        return []
    
    with open(filename, 'r') as file:
        urls = [line.strip() for line in file if line.strip()]
    
    return urls

# Main function to handle chat interaction
def main():
    filename = 'Sugarcane_links.txt'  # Adjust filename as needed
    urls = get_urls_from_file(filename)

    if not urls:
        print(f"No URLs found in {filename}. Exiting.")
        return

    print("Processing embeddings. This may take some time...")
    qa_chain = initialize_qa_chain(urls)
    print("Embeddings processed. You can now ask questions about the URLs.")

    # Chat loop
    while True:
        prompt = input("\nYou: ")
        if prompt.lower() in ["exit", "quit", "bye"]:
            print("Exiting the chatbot.")
            break
        response = process_answer(prompt, qa_chain)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
