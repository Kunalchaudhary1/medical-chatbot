from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from pinecone import Pinecone
import pinecone
from langchain.prompts import PromptTemplate

from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
import google.generativeai as genai  # Correct import
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
app = Flask(__name__)

load_dotenv()
os.environ["GOOGLE_API_KEY"] = "AIzaSyAb4j3-qvW93JyIw6-qqbn45_AyIB7MUoI"
api_key = os.environ["GOOGLE_API_KEY"]

genai.configure(api_key=api_key)


PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

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

docsearch= FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

#Loading the index



PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

# llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
#                   model_type="llama",
#                   config={'max_new_tokens':512,
#                           'temperature':0.8})

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.8,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key = "AIzaSyAb4j3-qvW93JyIw6-qqbn45_AyIB7MUoI"
)

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)


