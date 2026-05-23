from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import (GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI)
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
import os, tempfile
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def ingest_pdf(uploaded_file) -> FAISS:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", " "])
    chunks = splitter.split_documents(pages)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    
    return vector_store

def answer_question(vector_store: FAISS, question: str) -> dict:
    prompt_template = """
    You are a helpful assistant that helps in answering questions about a PDF document. Use ONLY the context
    provided below to answer. If the answer is not in the context, say "I coudn't find it in the document."

    Context: {context}
    Question: {input}
    Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    qa_chain = create_retrieval_chain(retriever, document_chain)

    result = qa_chain.invoke({
        "input": question
    })

    pages = sorted(list({
        doc.metadata.get("page", 0) + 1 for doc in result["context"]
    }))

    return {
        "answer": result["answer"],
        "source_pages": pages
    }