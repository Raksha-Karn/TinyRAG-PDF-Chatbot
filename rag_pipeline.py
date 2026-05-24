from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import (GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI)
from langchain_classic.chains import (create_retrieval_chain, create_history_aware_retriever)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
import os, tempfile
import json

CHAT_DIR = Path("chat_history")
CHAT_DIR.mkdir(exist_ok=True)

CHROMA_DIR = Path("chroma_db")
CHROMA_DIR.mkdir(exist_ok=True)

def save_chat(session_id, messages):
    filepath = CHAT_DIR / f"{session_id}.json"
    with open(filepath, "w") as f:
        json.dump(messages, f)

def load_chat(session_id):
    filepath = CHAT_DIR / f"{session_id}.json"
    if filepath.exists():
        with open(filepath, "r") as f:
            return json.load(f)
    return []

def ingest_pdf(uploaded_file, api_key: str, file_hash: str) -> Chroma:
    persist_dir = CHROMA_DIR / file_hash

    embeddings = (GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key))

    if persist_dir.exists():
        return Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150, separators=["\n\n", "\n", ".", " "])
    chunks = splitter.split_documents(pages)
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=str(persist_dir))
    
    return vector_store

def build_rag_chain(vector_store, api_key):
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0, google_api_key=api_key)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    contextualize_q_system_prompt = (
        "Given the chat history and latest "
        "user question, formulate a standalone "
        "question that can be understood without "
        "the chat history. Do NOT answer the "
        "question."
    )
    contextualize_q_prompt = (
        ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            (
                "human",
                "{input}"
            ),
        ])
    )
    history_aware_retriever = (
        create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    )

    qa_system_prompt = (
        "You are a helpful assistant for "
        "question-answering tasks. \n\n"
        "Use ONLY the retrieved context below "
        "to answer the question.\n\n"
        "If the answer is not in the context, "
        "say:\n"
        "'I couldn't find it in the document.'\n\n"
        "{context}"
    )

    qa_prompt = (
        ChatPromptTemplate.from_messages([
            (
                "system",
                qa_system_prompt
            ),
            MessagesPlaceholder(
                "chat_history"
            ),
            (
                "human",
                "{input}"
            ),
        ])
    )

    question_answer_chain = (create_stuff_documents_chain(llm, qa_prompt))
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

def format_chat_history(messages):
    chat_history = []
    for msg in messages:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))

    return chat_history

def answer_question(rag_chain, question, messages) -> dict:
    chat_history = format_chat_history(messages)
    result = rag_chain.invoke({
        "input": question,
        "chat_history": chat_history
    })

    pages = sorted(list({
        doc.metadata.get("page", 0) + 1 for doc in result["context"]
    }))

    return {
        "answer": result["answer"],
        "source_pages": pages
    }