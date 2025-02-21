from flask import Flask, render_template, request
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Initialize Flask App
app = Flask(__name__)

# Retrieve API Keys from Environment Variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")

if not GOOGLE_API_KEY or not PINECONE_API_KEY or not PINECONE_API_ENV:
    raise ValueError("Missing required environment variables! Set GOOGLE_API_KEY, PINECONE_API_KEY, and PINECONE_API_ENV.")

# Configure Google Generative AI
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.8,
    max_tokens=512,
    timeout=15,
    max_retries=1,
    api_key=GOOGLE_API_KEY
)

# Load Embeddings Model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-bot"

# Ensure index exists in Pinecone
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=1024, metric="cosine")

# Fix FAISS Loading Issue
FAISS_INDEX_PATH = "faiss_index"

if os.path.exists(FAISS_INDEX_PATH):
    try:
        docsearch = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print("Error loading FAISS index. Deleting and recreating index...", e)
        os.remove(FAISS_INDEX_PATH)
        docsearch = None
else:
    docsearch = None

if not docsearch:
    docsearch = FAISS.from_texts(["default text"], embeddings)
    docsearch.save_local(FAISS_INDEX_PATH)

# Create retriever
retriever = docsearch.as_retriever(search_kwargs={'k': 3})

# Define Prompt Template
PROMPT_TEMPLATE = """You are a medical chatbot. Use the following context to answer the question:
Context: {context}
Question: {question}
Answer:"""
PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

# Setup RetrievalQA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": PROMPT}
)

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
    port = int(os.environ.get("PORT", 8080))  # Render dynamically assigns PORT
    app.run(host="0.0.0.0", port=port, debug=True)
