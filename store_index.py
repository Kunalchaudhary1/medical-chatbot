from src.helper import load_pdf, text_split, download_hugging_face_embeddings

from pinecone import Pinecone
from dotenv import load_dotenv
from pinecone import ServerlessSpec
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os


from langchain_community.vectorstores import FAISS  # Updated import
import google.generativeai as genai  # Correct import
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import GPT2TokenizerFast

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings



load_dotenv()
os.environ["GOOGLE_API_KEY"] = "AIzaSyAb4j3-qvW93JyIw6-qqbn45_AyIB7MUoI"
api_key = os.environ["GOOGLE_API_KEY"]

genai.configure(api_key=api_key)

# Initialize the tokenizer for token counting (from transformers)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
# print(text_chunks)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)


index_name="medical-bot"

# Check if index already exists
existing_indexes = pc.list_indexes().names()

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1024,  # Ensure this matches your embedding model output
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print(f"Index '{index_name}' already exists. Skipping creation.")


#Creating Embeddings for Each of The Text Chunks & storing
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.from_documents(text_chunks, embeddings)

# Save the FAISS index locally
db.save_local("faiss_index")  # Save to a local folder, adjust as needed

# Later, load the FAISS index
db_loaded = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)