
import streamlit as st
import uuid
import json
import os

import crag
import ingestion


st.set_page_config(
    page_title="Corrective RAG",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Corrective RAG – Ask Your Documents")


# ---------- Persistent Chat History ----------

CHAT_FILE = "chat_history.json"


def load_chats():
    if os.path.exists(CHAT_FILE):
        try:
            with open(CHAT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data["conversations"], data["active_chat_id"]
        except (json.JSONDecodeError, KeyError):
            pass

    chat_id = str(uuid.uuid4())

    return [
        {
            "id": chat_id,
            "title": "New Chat",
            "history": []
        }
    ], chat_id


def save_chats():
    data = {
        "conversations": st.session_state.conversations,
        "active_chat_id": st.session_state.active_chat_id
    }

    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if "conversations" not in st.session_state:
    conversations, active_chat_id = load_chats()
    st.session_state.conversations = conversations
    st.session_state.active_chat_id = active_chat_id


# ---------- Sidebar ----------

with st.sidebar:

    st.header("💬 Chat History")

    if st.button("➕ New Chat", use_container_width=True):
        chat_id = str(uuid.uuid4())

        st.session_state.conversations.insert(
            0,
            {
                "id": chat_id,
                "title": "New Chat",
                "history": []
            }
        )

        st.session_state.active_chat_id = chat_id
        save_chats()
        st.rerun()

    for chat in st.session_state.conversations:

        if st.button(
            chat["title"],
            key=f"select_{chat['id']}",
            use_container_width=True
        ):
            st.session_state.active_chat_id = chat["id"]
            save_chats()
            st.rerun()

    st.divider()

    st.header("📄 Documents")

    files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True
    )

    if st.button("Process files") and files:

        for f in files:

            with st.spinner(f"Processing {f.name}..."):

                try:
                    n = ingestion.ingest_file(
                        f.name,
                        f.getvalue()
                    )

                    st.success(f"{f.name}: {n} chunks")

                except Exception as e:
                    st.error(f"{f.name}: {e}")

    st.markdown("**Current documents:**")

    for d in ingestion.list_documents():
        st.write("•", d)

    if st.button("🗑️ Clear database"):

        ingestion.reset_db()

        for chat in st.session_state.conversations:
            chat["history"] = []

        save_chats()
        st.rerun()


# ---------- Render Answer ----------

def render(result):

    st.markdown(result["answer"])

    if result["sources"]:

        with st.expander("📚 Sources"):

            for i, s in enumerate(result["sources"], 1):

                st.markdown(
                    f"**[{i}] {s['source']} – page {s['page']}**"
                )

                st.caption(s["text"])

    with st.expander("🧭 Processing steps (Trace)"):

        for t in result["trace"]:
            st.write("-", t)


# ---------- Active Chat ----------

active_chat = next(
    chat for chat in st.session_state.conversations
    if chat["id"] == st.session_state.active_chat_id
)


# ---------- Display Previous Messages ----------

for item in active_chat["history"]:

    with st.chat_message("user"):
        st.write(item["q"])

    with st.chat_message("assistant"):
        render(item["r"])


# ---------- Ask Question ----------

question = st.chat_input("Type your question here...")

if question:

    with st.spinner("Searching and evaluating results..."):

        result = crag.answer(question)

    active_chat["history"].append({
        "q": question,
        "r": result
    })

    if active_chat["title"] == "New Chat":

        active_chat["title"] = question[:35]

        if len(question) > 35:
            active_chat["title"] += "..."

    save_chats()
    st.rerun()
