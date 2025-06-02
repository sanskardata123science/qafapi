import os 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
import streamlit as st
from q.endpoint import FastAPI, HTTPException
from pydantic import BaseModel



load_dotenv()
loader = TextLoader(r"C:\Users\sansk\OneDrive\Documents\chatbtqa\carroll-alice.txt")
docs = loader.load()


splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(chunks, embeddings)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
retriever = vectorstore.as_retriever()

rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
system_prompt = (
    "you are a helpful assistant answer only thode question that are based on document"
    "if question is not related to document respond with: "
    "\"I'm sorry, that information is not available in this document.\""
)
def format_query(que):
    return f"{system_prompt}\n\nQuestion: {que}"
app = FastAPI()
class Query(BaseModel):
    question : str
@app.get("/")
def read_root():
    return {"message":"alice in the wonderlad q/a api is runninng"}
@app.post("/ask")
def ask_que(query:Query):
    if not query.question:
        raise HTTPException(status_code=400, detail="Question is required.")
    formatted_query = format_query(query.question)
    answer = rag_chain.run(formatted_query)
    return {"answer" : answer}



