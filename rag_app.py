from __future__ import annotations

import os

import streamlit as st

from rag_agent import RagAgent, RetrievalHit


st.set_page_config(
    page_title="Tennis RAG Coach",
    page_icon="🎾",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=Fraunces:ital,wght@0,700;0,900;1,700&display=swap');

    /* ── Reset & base ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: #1A2E1E;
    }
    .stApp {
        background-color: #F7F9F4;
        color: #1A2E1E;
    }
    .stApp p, .stApp span, .stApp div, .stApp li {
        color: #1A2E1E;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #1B4332 !important;
        border-right: 3px solid #C8E32B;
    }
    [data-testid="stSidebar"] * {
        color: #E8F5E9 !important;
    }
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stFileUploader label,
    [data-testid="stSidebar"] .stTextArea label {
        color: #A5D6A7 !important;
        font-size: 0.75rem !important;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    /* Sidebar title */
    [data-testid="stSidebar"] h1 {
        font-family: 'Fraunces', serif !important;
        font-size: 1.5rem !important;
        color: #C8E32B !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.1rem;
    }
    [data-testid="stSidebar"] .stCaption {
        color: #81C784 !important;
    }

    /* Sidebar metrics */
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background-color: rgba(255,255,255,0.07);
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        border: 1px solid rgba(200, 227, 43, 0.2);
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #C8E32B !important;
        font-weight: 700;
        font-size: 1.5rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #A5D6A7 !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* Sidebar subheader */
    [data-testid="stSidebar"] h3 {
        color: #C8E32B !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
        margin-top: 0.5rem;
        margin-bottom: 0.25rem;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background-color: transparent !important;
        border: 1px solid rgba(200, 227, 43, 0.4) !important;
        color: #C8E32B !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500;
        font-size: 0.8rem;
        border-radius: 6px;
        transition: all 0.15s ease;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: rgba(200, 227, 43, 0.1) !important;
        border-color: #C8E32B !important;
    }

    /* Sidebar form submit button */
    [data-testid="stSidebar"] .stFormSubmitButton > button {
        background-color: #C8E32B !important;
        border: none !important;
        color: #1B4332 !important;
        font-weight: 700;
        font-family: 'DM Sans', sans-serif !important;
        border-radius: 6px;
        transition: all 0.15s ease;
    }
    [data-testid="stSidebar"] .stFormSubmitButton > button:hover {
        background-color: #d8f03b !important;
    }

    /* Sidebar divider */
    [data-testid="stSidebar"] hr {
        border-color: rgba(200, 227, 43, 0.2) !important;
    }

    /* Sidebar info box */
    [data-testid="stSidebar"] .stAlert {
        background-color: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(200, 227, 43, 0.2) !important;
        color: #A5D6A7 !important;
        border-radius: 8px;
    }

    /* ── Main area title ── */
    .main-title {
        font-family: 'Fraunces', serif;
        font-size: 2.6rem;
        font-weight: 900;
        color: #1B4332;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 0;
    }
    .main-subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.9rem;
        color: #6B8F71;
        margin-top: 0.4rem;
        margin-bottom: 1.25rem;
    }

    /* ── Suggestion chips ── */
    .suggestion-block {
        background: #FFFFFF;
        border: 1px solid #D8E8DA;
        border-left: 3px solid #C8E32B;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
        font-size: 0.88rem;
        color: #2D5A3D;
        line-height: 1.5;
    }
    .suggestion-block strong {
        color: #1B4332;
        font-weight: 600;
        display: block;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.35rem;
        color: #7BAF85;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: #FFFFFF !important;
        border: 1px solid #E4EEE6 !important;
        border-radius: 12px !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.75rem !important;
        box-shadow: 0 1px 4px rgba(27, 67, 50, 0.06);
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] em {
        color: #1A2E1E !important;
    }
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: #EEF7F0 !important;
        border-color: #C8E32B !important;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] {
        border-radius: 12px !important;
        border: 2px solid #1B4332 !important;
        background: #FFFFFF !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #C8E32B !important;
        box-shadow: 0 0 0 3px rgba(200, 227, 43, 0.2) !important;
    }

    /* ── Expander (sources) ── */
    .streamlit-expanderHeader {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        color: #2D5A3D !important;
        background: #F0F7F1 !important;
        border-radius: 8px !important;
        border: 1px solid #D8E8DA !important;
        letter-spacing: 0.03em;
    }
    .streamlit-expanderContent {
        border: 1px solid #D8E8DA !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        background: #FAFDFB !important;
    }

    /* ── Info/error alerts in main ── */
    .stAlert {
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* ── Headings in main ── */
    h1 { font-family: 'Fraunces', serif !important; color: #1B4332; }
    h2, h3 { font-family: 'DM Sans', sans-serif !important; color: #2D5A3D; font-weight: 600; }

    /* ── Spinner text ── */
    .stSpinner > div { color: #1B4332 !important; }

    /* ── Password input ── */
    [data-testid="stSidebar"] input[type="password"] {
        background-color: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(200, 227, 43, 0.3) !important;
        color: #E8F5E9 !important;
        border-radius: 6px;
    }
    [data-testid="stSidebar"] input[type="password"]::placeholder {
        color: rgba(232, 245, 233, 0.4) !important;
    }

    /* ── File uploader ── */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background-color: rgba(255,255,255,0.06) !important;
        border: 1px dashed rgba(200, 227, 43, 0.35) !important;
        border-radius: 8px;
        padding: 0.5rem;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #F0F7F1; }
    ::-webkit-scrollbar-thumb { background: #A5C4AC; border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_agent() -> RagAgent:
    if "agent" not in st.session_state:
        st.session_state.agent = RagAgent()
    return st.session_state.agent


def load_uploaded_files(agent: RagAgent, uploads: list) -> None:
    processed_names = st.session_state.setdefault("processed_uploads", set())
    for upload in uploads:
        file_key = (upload.name, upload.size)
        if file_key in processed_names:
            continue
        file_bytes = upload.getvalue()
        if upload.name.lower().endswith(".pdf"):
            agent.add_pdf_document(upload.name, file_bytes)
        else:
            content = file_bytes.decode("utf-8", errors="ignore")
            agent.add_text_document(upload.name, content, mime_type=upload.type)
        processed_names.add(file_key)


def add_pasted_text(agent: RagAgent, pasted_text: str) -> None:
    if not pasted_text.strip():
        return
    paste_index = len(agent.documents) + 1
    agent.add_text_document(
        name=f"pasted-note-{paste_index}.txt",
        content=pasted_text.strip(),
        mime_type="text/plain",
    )


def render_sidebar(agent: RagAgent) -> str:
    with st.sidebar:
        st.title("🎾 AI Tennis")
        st.caption("Beginner Q&A powered by your tennis documents")

        st.divider()

        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.get("api_key", os.getenv("GEMINI_API_KEY", "")),
            placeholder="Paste your key here",
            help="Used only to call the Gemini API for answers.",
        )
        st.session_state.api_key = api_key

        st.divider()

        uploads = st.file_uploader(
            "Upload Tennis Documents",
            type=["txt", "md", "csv", "json", "py", "js", "ts", "html", "css", "pdf"],
            accept_multiple_files=True,
        )
        if uploads:
            load_uploaded_files(agent, uploads)

        with st.form("paste_tennis_content", clear_on_submit=True):
            pasted_text = st.text_area(
                "Paste Tennis Notes",
                height=120,
                placeholder="Paste rules, drills, match notes, strategy guides…",
            )
            submitted = st.form_submit_button("Add Notes →", use_container_width=True)
        if submitted:
            add_pasted_text(agent, pasted_text)
            st.rerun()

        st.divider()

        stats = agent.stats()
        col1, col2 = st.columns(2)
        col1.metric("Docs", stats["documents"])
        col2.metric("Chunks", stats["chunks"])
        st.metric("Questions Asked", stats["questions"])

        st.subheader("Knowledge Base")
        if not agent.documents:
            st.info("No documents loaded yet.")
        else:
            for document in agent.documents:
                label = f"📄 {document.name} · {document.chunk_count} chunks"
                if document.is_pdf and document.page_count:
                    label += f" · {document.page_count}pp"
                st.caption(label)

        st.divider()

        col3, col4 = st.columns(2)
        if col3.button("Clear Chat", use_container_width=True):
            agent.clear_chat()
            st.rerun()
        if col4.button("Clear Docs", use_container_width=True):
            agent.clear_documents()
            st.session_state.processed_uploads = set()
            st.rerun()

    return api_key


def render_sources(hits: list[RetrievalHit]) -> None:
    if not hits:
        return
    with st.expander(f"📎 {len(hits)} source chunks retrieved"):
        for index, hit in enumerate(hits, start=1):
            page_suffix = f" · p.{hit.page}" if hit.page else ""
            st.markdown(
                f"**[{index}] {hit.document_name}{page_suffix}** &nbsp; `score {hit.score:.3f}`"
            )
            st.caption(hit.text[:500] + ("…" if len(hit.text) > 500 else ""))
            if index < len(hits):
                st.divider()


def render_chat(agent: RagAgent) -> None:
    st.markdown('<p class="main-title">Learn Tennis, Ask Anything</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="main-subtitle">'
        "Upload your tennis guides, rules, or drills — then ask beginner questions grounded in your documents."
        "</p>",
        unsafe_allow_html=True,
    )

    if not agent.documents:
        suggestions = [
            ("How does tennis scoring work?", "Scoring"),
            ("What is the difference between a let and a fault?", "Rules"),
            ("How should a beginner hit a forehand?", "Technique"),
            ("What are the main rules for serving?", "Serving"),
            ("What is the best strategy on clay for a beginner?", "Strategy"),
            ("List confusing tennis terms and define them simply.", "Vocabulary"),
        ]
        st.markdown(
            '<div class="suggestion-block">'
            "<strong>Example questions to try</strong>"
            + " &nbsp;·&nbsp; ".join(f"<em>{q}</em>" for q, _ in suggestions)
            + "</div>",
            unsafe_allow_html=True,
        )
        st.info("⬅️ Upload tennis guides, match notes, rules, drills, or paste text in the sidebar to get started.")
        return

    for message in agent.chat_history:
        role = "🎾" if message["role"] == "assistant" else "🧑"
        with st.chat_message(role):
            st.markdown(message["content"])


def main() -> None:
    agent = get_agent()
    api_key = render_sidebar(agent)
    render_chat(agent)

    prompt = st.chat_input("Ask a beginner tennis question…")
    if not prompt:
        return

    with st.chat_message("🧑"):
        st.markdown(prompt)

    if not api_key:
        with st.chat_message("🎾"):
            st.error("Add your Gemini API key in the sidebar first.")
        return

    if not agent.documents:
        with st.chat_message("🎾"):
            st.error("Upload at least one tennis document before asking a question.")
        return

    with st.chat_message("🎾"):
        with st.spinner("Finding relevant context and generating answer…"):
            try:
                answer, hits = agent.answer_question(prompt, api_key=api_key)
            except Exception as exc:
                st.error(str(exc))
                return
            st.markdown(answer)
            render_sources(hits)


if __name__ == "__main__":
    main()
