from flask import Flask, render_template, request
import os
import shutil
import pickle

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pinecone import Pinecone

# Initialize Flask App
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Set API Keys (Do NOT hardcode API keys)
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing from environment variables!")

os.environ["GOOGLE_API_KEY"] = api_key

# Configure Google Generative AI
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.8,
    max_tokens=512,  # Restrict response length for better latency
    timeout=15,  # Reduce API call timeout
    max_retries=1  # Reduce retries to avoid unnecessary delay
)

# Load Embeddings Model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")

if not PINECONE_API_KEY or not PINECONE_API_ENV:
    raise ValueError("PINECONE_API_KEY or PINECONE_API_ENV is missing!")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-bot"

# Ensure Pinecone index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine"
    )

# FAISS Index Path
FAISS_INDEX_PATH = "faiss_index"

# Function to load FAISS index safely
def load_faiss():
    try:
        print("Loading FAISS index...")
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    except (KeyError, EOFError, pickle.UnpicklingError) as e:
        print(f"Error loading FAISS index: {e}")
        print("Deleting and recreating index...")

        # Remove corrupted index safely
        if os.path.isdir(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH)
        elif os.path.exists(FAISS_INDEX_PATH):
            os.remove(FAISS_INDEX_PATH)

        # Recreate FAISS index
        new_index = FAISS.from_documents([], embeddings)
        new_index.save_local(FAISS_INDEX_PATH)
        print("FAISS index recreated.")
        return new_index

# Load FAISS Index
docsearch = load_faiss()

# Create retriever
retriever = docsearch.as_retriever(search_kwargs={'k': 3})  # Optimized `k` value

# Define Prompt Template
PROMPT_TEMPLATE = """You are a medical chatbot. Use the following context to answer the question:
Context: {context}
Question: {question}
Answer:"""

PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

# Set up RetrievalQA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Using `stuff` to fix chain issues
    retriever=retriever,
    return_source_documents=False,  # Avoid unnecessary data transfer
    chain_type_kwargs={"prompt": PROMPT}
)

# Flask Routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print("User Input:", msg)

    # Query the LLM
    result = qa({"query": msg})

    response = result["result"]
    print("Response:", response)

    return str(response)

if __name__ == '__main__':
    port = int(os.getenv("PORT", 8080))  # Use Render's assigned PORT
    app.run(host="0.0.0.0", port=port, debug=True)
