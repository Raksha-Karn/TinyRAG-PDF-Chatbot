import hashlib
import streamlit as st
import uuid
from rag_pipeline import ingest_pdf, answer_question, build_rag_chain, load_chat, save_chat

st.set_page_config(page_title="PDF Q&A", page_icon="📄", layout="centered")
st.title("Chat with your PDF 📄")
st.caption("Upload a PDF and ask anything you want to know!")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state["messages"] = load_chat(st.session_state["session_id"])

with st.sidebar:
    st.header("Settings")
    user_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Your key stays only in your session."
    )
    st.header("Upload PDF")
    uploaded = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded:
        if uploaded.size > 10 * 1024 * 1024:
            st.error("PDF too large (max 10 MB)")
            st.stop()
        
        file_hash = hashlib.md5(uploaded.getvalue()).hexdigest()

        if st.session_state.get("file_hash") != file_hash:
            if st.button("Process PDF", type="primary"):
                if not user_api_key:
                    st.warning("Enter Gemini API Key")
                    st.stop()

                with st.spinner("Reading and embedding your PDF"):
                    vector_store = ingest_pdf(uploaded, user_api_key, file_hash)
                    chain = build_rag_chain(vector_store, user_api_key)
                    st.session_state["vector_store"] = vector_store
                    st.session_state["chain"] = chain
                    st.session_state["file_hash"] = file_hash
                    st.session_state["messages"] = []
                    
                    save_chat(st.session_state["session_id"], [])
                st.success("PDF Processed! Ask questions below.")
        else:
            st.info("This PDF is already processed!")

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask something about the PDF: "):
    if "chain" not in st.session_state:
        st.warning("Upload and process a PDF first!")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": question})
    save_chat(st.session_state["session_id"], st.session_state["messages"])

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking.."):
            result = answer_question(st.session_state["chain"], question, st.session_state["messages"])
        answer = result["answer"]
        st.markdown(answer)

        if result["source_pages"]:
            pages_str = ", ".join(map(str, result["source_pages"]))
            st.caption(f"Sources: page(s) {pages_str}")
            answer += (
                f"\n\n*Sources: "
                f"page(s) {pages_str}*"
            )        

    st.session_state["messages"].append({"role": "assistant", "content": answer})
    save_chat(st.session_state["session_id"], st.session_state["messages"])