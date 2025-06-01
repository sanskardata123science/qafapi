import os 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
import streamlit as st



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
query = "Who is the Queen of Hearts?"
print(rag_chain.run(query))

st.title("📚 Alice in Wonderland Q&A")

query = st.text_input("Ask a question:")

if query:
    answer = rag_chain.run(query)
    st.write("💬 Answer:", answer)
