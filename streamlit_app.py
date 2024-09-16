import json
import os
import sys
import boto3
import tempfile
import streamlit as st

from langchain_aws import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Set page configuration at the very top
st.set_page_config(page_title="Chat with PDF using AWS Bedrock", page_icon="🌍")

# Access AWS credentials from Streamlit secrets
aws_access_key_id = st.secrets["aws"]["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"]
aws_region = st.secrets["aws"]["AWS_DEFAULT_REGION"]

## Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime",
                        region_name=aws_region,
                        aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Apply custom CSS for styling
st.markdown("""
    <style>
        .main-header {
            font-size: 36px;
            font-weight: bold;
            color: #34A853;
            text-align: center;
            margin-top: 20px;
        }
        .sub-header {
            font-size: 22px;
            color: #0B5394;
            margin-top: 20px;
            text-align: center;
        }
        .sidebar-button {
            background-color: #E0F7FA;
            color: #01579B;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px;
            border: 1px solid #0288D1;
            transition: background-color 0.3s ease;
        }
        .sidebar-button:hover {
            background-color: #0288D1;
            color: white;
        }
        .success-message {
            color: #34A853;
        }
        .description {
            font-size: 18px;
            color: #333;
            margin-top: 10px;
            background-color: #F1F8E9;
            padding: 10px;
            border-radius: 5px;
        }
        .footer {
            font-size: 14px;
            color: #888;
            text-align: center;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

## Data ingestion
def data_ingestion(uploaded_files=None):
    if uploaded_files:
        # Load user-uploaded PDFs
        documents = []
        for uploaded_file in uploaded_files:
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            
            # Load the PDF using PyPDFLoader with the temporary file path
            loader = PyPDFLoader(temp_file_path)
            documents.extend(loader.load())

            # Optionally delete the temporary file after loading it
            os.remove(temp_file_path)

    return documents

def get_titan_llm():
    ## Create the Amazon Titan model
    llm = Bedrock(
        model_id="amazon.titan-text-express-v1",
        client=bedrock,
        model_kwargs={
            'maxTokenCount': 8192,
            'temperature': 0,
            'topP': 1
        }
    )
    return llm

prompt_template = """
Human: Using the provided context, deliver a comprehensive 
and concise answer to the question that follows. Ensure the 
response is a minimum of 250 words, offering detailed explanations. 
If the information is not available, clearly state that the answer 
is unknown rather than attempting to fabricate a response.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# def preprocess_query(query):
#     # Simplify the query by removing common question words
#     query = query.lower().strip()
#     if query.startswith("what is "):
#         query = query.replace("what is ", "")
#     return query

def get_response_llm(llm, vectorstore_faiss, query):
    # Preprocess the query before retrieving the answer
    # query = preprocess_query(query)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']


def main():
    # Initialize session state for faiss_index
    if 'faiss_index' not in st.session_state:
        st.session_state['faiss_index'] = None

    # Main Header
    st.markdown('<div class="main-header">🌍 Chat with PDF using AWS Bedrock 🧠</div>', unsafe_allow_html=True)
    
    st.write("Welcome to the PDF Chatbot! Use this app to interact with PDFs using Amazon Bedrock's AI models.")
    
    # Description of the default dataset
    st.markdown("""
    <div class="description">
        📚 Default Dataset: This dataset contains information on topics such as Greenhouse Gas (GHG) emissions, 
        Environmental, Social, and Governance (ESG) criteria, climate change policies, and related 
        environmental agreements. You can ask questions about subjects like the Kyoto Protocol, 
        the Paris Agreement, carbon footprint, climate initiatives, and more. 
        <br>💡 Example Questions: "What is the Kyoto Protocol?" or "What is NetZero?"
    </div>
    """, unsafe_allow_html=True)

    # Layout for File Upload and Vector Store Options
    st.sidebar.title("Options 🛠️")

    # Sidebar description for the default dataset
    st.sidebar.markdown("""
    You can upload your own PDFs and customize the chatbot!
    """)

    # File uploader for up to 5 PDFs
    uploaded_files = st.sidebar.file_uploader(
        "Upload up to 5 PDF files:", accept_multiple_files=True, type=['pdf'], help="You can upload up to 5 PDF files for vectorization."
    )

    # Limit the number of uploaded files to 5
    if uploaded_files and len(uploaded_files) > 5:
        st.sidebar.error("You can only upload up to 5 PDF files.")
        uploaded_files = uploaded_files[:5]

    # Separator line
    st.sidebar.markdown("---")

    # Separate buttons for updating vectors using uploaded files or default files
    with st.sidebar:
        st.markdown('<div class="sub-header">Update Vector Store</div>', unsafe_allow_html=True)
        st.write("Choose whether to use the uploaded PDFs or the default set.")
        st.button("📂 Update Vector with Uploaded PDFs", key="update_uploaded", help="Use this option to update the vector store with your uploaded PDFs.", on_click=lambda: update_vector(uploaded_files))
        st.button("📁 Use Precomputed Vector Store for Default PDFs", key="use_default", help="Use the precomputed vector store for the default dataset.", on_click=use_default_vector_store)

    # Separator line
    st.markdown("---")

    # User Input for Questions
    st.markdown('<div class="sub-header">Ask a Question from the PDF Files 📄</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="ask-question-section">', unsafe_allow_html=True)
        user_question = st.text_input("Enter your question:", placeholder="Type your question here...")

        # Button to get the response from the LLM
        if st.button("💬 Get Answer"):
            if st.session_state['faiss_index'] is None:
                st.error("Please select or upload PDFs and update the vector store before asking a question.")
            else:
                with st.spinner("Processing your question..."):
                    try:
                        llm = get_titan_llm()
                        answer = get_response_llm(llm, st.session_state['faiss_index'], user_question)
                        st.success("📜 Here is the answer:")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"Error during LLM response generation: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown('<div class="footer">🌿 Empowering Environmental Awareness with AI 🌱</div>', unsafe_allow_html=True)

def update_vector(uploaded_files):
    if not uploaded_files:
        st.sidebar.warning("Please upload some PDF files first.")
    else:
        with st.spinner("Processing uploaded PDFs..."):
            docs = data_ingestion(uploaded_files)
            # Check if docs are empty
            if not docs:
                st.error("No text found in the uploaded PDFs. Please check the files and try again.")
                st.stop()
            else:
                # Create and save a new vector store for the uploaded files
                try:
                    st.session_state['faiss_index'] = FAISS.from_documents(docs, bedrock_embeddings)
                    st.session_state['faiss_index'].save_local("faiss_index")
                    st.success("Vector store updated with uploaded PDFs.", icon="✅")
                except ValueError as e:
                    st.error(f"Error creating vector store: {e}")

def use_default_vector_store():
    with st.spinner("Loading precomputed FAISS index for default PDFs..."):
        st.session_state['faiss_index'] = FAISS.load_local("default_faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
        st.success("Loaded precomputed vector store for default PDFs.", icon="✅")

if __name__ == "__main__":
    main()
