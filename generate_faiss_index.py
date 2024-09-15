# generate_faiss_index.py
import os
import boto3
import tempfile
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

## Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime", region_name='us-east-1')
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

def generate_default_faiss_index():
    # Load the default set of PDFs
    loader = PyPDFDirectoryLoader("data")  # Adjust the path if needed
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)

    # Create the FAISS vector store
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)

    # Save the FAISS index to a file
    vectorstore_faiss.save_local("default_faiss_index")
    print("FAISS index for the default dataset has been generated and saved.")

if __name__ == "__main__":
    generate_default_faiss_index()
