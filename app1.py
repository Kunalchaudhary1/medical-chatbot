from flask import Flask, render_template, request
import os
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

# Set API Keys
os.environ["GOOGLE_API_KEY"] = "AIzaSyAb4j3-qvW93JyIw6-qqbn45_AyIB7MUoI"
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Google Generative AI
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.8,
    max_tokens=512,  # Restrict response length for better latency
    timeout=15,  # Reduce API call timeout
    max_retries=1,  # Reduce retries to avoid unnecessary delay
    api_key=api_key
)

# Load Embeddings Model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-bot"

# Ensure index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine"
    )

# Load FAISS Index
docsearch = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Create retriever
retriever = docsearch.as_retriever(search_kwargs={'k': 3})  # Optimized k-value

# Define Prompt Template
PROMPT_TEMPLATE = """You are a medical chatbot. Use the following context to answer the question:
Context: {context}
Question: {question}
Answer:"""
PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

# Set up RetrievalQA Chain (Using `stuff` instead of `map_reduce`)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Switched to `stuff` to fix the error
    retriever=retriever,
    return_source_documents=False,  # Avoid unnecessary data transfer
    chain_type_kwargs={"prompt": PROMPT}  # Correct way to pass the prompt
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
    app.run(host="0.0.0.0", port=8080, debug=True)
