import os
import streamlit as st
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# --- 1. AUTHENTICATION ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.set_page_config(page_title="Login | PartsCheck", page_icon="🔒")
    st.markdown("# 🔒 PartsCheck Support Hub")
    password = st.text_input("Enter Password to Access", type="password")
    if st.button("Log In"):
        if password == st.secrets.get("APP_PASSWORD", "PartsCheck2026"):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# --- 2. CONFIG & DATA ---
CSV_PATH = "processed_docs/final_rag_dataset.csv"
IMG_DIR = "images"

@st.cache_resource
def load_data():
    if not os.path.exists(CSV_PATH):
        st.error(f"CSV file not found.")
        st.stop()
    df = pd.read_csv(CSV_PATH, encoding='utf-8').dropna(subset=['process', 'sub_process'])
    unique_subs = df[['process', 'sub_process']].drop_duplicates()
    search_docs = [Document(page_content=f"{r['process']} {r['sub_process']}",
                   metadata={'sub': r['sub_process']}) for _, r in unique_subs.iterrows()]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(search_docs, embeddings)
    return vectorstore, df, unique_subs['sub_process'].tolist()

vectorstore, df, all_sub_processes = load_data()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=st.secrets["GROQ_API_KEY"],
    temperature=0
)

# --- 3. SESSION STATE ---
if "target_sub" not in st.session_state:
    st.session_state.target_sub = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def select_procedure(name):
    st.session_state.target_sub = name

# --- 4. UI LAYOUT ---
st.set_page_config(page_title="PartsCheck Assistant", layout="wide")

with st.sidebar:
    st.markdown("👤 **User:** PartsCheck Member")
    st.markdown("---")
    if st.button("➕ Start New Inquiry", use_container_width=True):
        st.session_state.target_sub = None
        st.session_state.chat_history = []
        st.rerun()
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

st.markdown("# 🛠️ PartsCheck Smart Assistant")

# --- 5. DISPLAY AREA ---
if st.session_state.target_sub:
    target = st.session_state.target_sub
    st.success(f"### SOP Guide: {target}")
    steps = df[df['sub_process'] == target].sort_values('step_counter')
    for _, row in steps.iterrows():
        with st.container(border=True):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader(f"Step {row['step_counter']}")
                st.write(row['text'])
            with c2:
                img_name = os.path.basename(str(row.get('image_path', '')))
                path = os.path.join(IMG_DIR, img_name)
                if os.path.exists(path):
                    st.image(path, use_container_width=True)
    if st.button("❌ Close Guide"):
        st.session_state.target_sub = None
        st.rerun()
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# --- 6. IMPROVED ROUTING (CSV FIRST) ---
if question := st.chat_input("Ask about an SOP..."):
    st.session_state.target_sub = None
    with st.chat_message("user"):
        st.write(question)
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        # STEP 1: Check CSV via Vector Search
        # We look for matches with a high similarity
        matches = vectorstore.similarity_search_with_score(question, k=3)
        
        # Filter matches by a "confidence" score (lower score = more similar)
        # 0.6 is a good balance for MiniLM embeddings
        relevant_matches = [m for m, score in matches if score < 1.0] 

        if relevant_matches:
            st.markdown("### 📄 Related Procedures Found:")
            st.info("Click a procedure below to view the official steps and images.")
            for i, match in enumerate(relevant_matches):
                sub_name = match.metadata['sub']
                st.button(f"View: {sub_name}", key=f"opt_{i}", on_click=select_procedure, args=(sub_name,))
        else:
            # STEP 2: Only if no CSV match, then use LLM
            chat_prompt = f"""
            You are the PartsCheck SaaS Expert. 
            Tone: Professional, Corporate, subtly Australian.
            Constraint: If you don't know the exact steps for a PartsCheck process, 
            advise the user to contact support or try different keywords.
            User Question: {question}
            """
            response = llm.invoke(chat_prompt).content
            st.write(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
