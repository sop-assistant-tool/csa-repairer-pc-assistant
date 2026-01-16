import os
import streamlit as st
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIG & DATA ---
CSV_PATH = "processed_docs/final_rag_dataset.csv"
IMG_DIR = "processed_docs/images"

@st.cache_resource
def load_data():
    if not os.path.exists(CSV_PATH):
        st.error(f"CSV file not found at: {CSV_PATH}. Please check the path and redeploy.")
        st.stop()
    try:
        df = pd.read_csv(CSV_PATH, encoding='utf-8').dropna(subset=['process', 'sub_process'])
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()

    unique_subs = df[['process', 'sub_process']].drop_duplicates()
    search_docs = [Document(page_content=f"{r['process']} {r['sub_process']}",
                   metadata={'sub': r['sub_process']}) for _, r in unique_subs.iterrows()]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(search_docs, embeddings)
    return vectorstore, df, unique_subs['sub_process'].tolist()

vectorstore, df, all_sub_processes = load_data()

# --- SESSION STATE ---
if "target_sub" not in st.session_state:
    st.session_state.target_sub = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def select_procedure(name):
    st.session_state.target_sub = name
    st.rerun()

# --- UI LAYOUT ---
st.set_page_config(page_title="PartsCheck Assistant", layout="wide")
st.markdown("# 🛠️ PartsCheck Smart Assistant")

with st.sidebar:
    logo_path = r"C:\Users\ErickMortera\Pictures\PartsCheck logo.png"  # Local path - won't work online, remove or host online
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.warning("Logo file not found.")

    st.markdown("---")
    if st.button("➕ Start New Inquiry", use_container_width=True):
        st.session_state.target_sub = None
        st.session_state.chat_history = []
        st.rerun()

# --- DISPLAY AREA ---
display_container = st.container()

with display_container:
    # Welcome Card (only if no history and no SOP selected)
    if not st.session_state.chat_history and not st.session_state.target_sub:
        with st.container(border=True):
            st.markdown("""
            ### Welcome to the PartsCheck Support Hub
            I am your dedicated **SaaS Support Assistant**. I can help you navigate the PartsCheck platform, 
            optimize your quoting workflow, and access official Standard Operating Procedures (SOPs).
            
            **How to use this assistant:**
            * **Search SOPs:** Ask things like *"How do I create a quote?"* or *"How do I receipt parts?"*
            * **General Help:** Ask about platform features or industry standards.
            * **Quick Actions:** Use the sidebar to reset the session at any time.
            
            *Tip: If I find multiple matching procedures, I'll provide buttons for you to choose the right one.*
            """)
            st.info("💡 **Try asking:** 'How do I integrate with iBodyShop?'")

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
                    else:
                        st.caption(f"*(Visual help missing: {img_name})*")
        
        if st.button("❌ Close Guide & Return to Chat"):
            st.session_state.target_sub = None
            st.rerun()

    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

# --- INPUT & ROUTING ---
if question := st.chat_input("Ask about an SOP or the industry..."):
    # Always clear the SOP guide when a new question is typed
    st.session_state.target_sub = None
    
    with st.chat_message("user"):
        st.write(question)
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.status("🔍 Analyzing request...", expanded=True) as status:
            # Simple keyword-based routing (no LLM needed)
            question_lower = question.lower()
            matched_sub = None

            # Keyword matching for sub_processes (expand as needed)
            if 'receipt' in question_lower or 'invoice' in question_lower:
                if 'ibodyshop' in question_lower or 'body shop' in question_lower:
                    matched_sub = 'Receipting in IBodyShop'
                elif 'flexiquote cloud' in question_lower or 'cloud' in question_lower:
                    matched_sub = 'Receipting in Flexiquote Cloud'
                elif 'flexiquote desktop' in question_lower or 'desktop' in question_lower:
                    matched_sub = 'Receipting in Flexiquote Desktop'
            elif 'add supplier' in question_lower or 'assign supplier' in question_lower:
                matched_sub = 'Adding a Supplier'
            elif 'remove supplier' in question_lower or 'delete supplier' in question_lower:
                matched_sub = 'Removing a Supplier'
            elif 'tier' in question_lower or 'tiering' in question_lower:
                matched_sub = 'Tiering Suppliers'
            elif 'integrate' in question_lower or 'integration' in question_lower:
                matched_sub = 'Integrating PartsCheck'
            elif 'report' in question_lower or 'reporting' in question_lower:
                matched_sub = 'Repairer Reporting'
            # Add more keyword checks for other sub_processes

            status.update(label="Intent Identified!", state="complete", expanded=False)

        if matched_sub:
            st.session_state.target_sub = matched_sub
            st.rerun()
        else:
            st.info("No exact SOP match found. Try being more specific (e.g. 'receipting in ibodyshop' or 'add supplier').")
            # Optional: List all sub_processes as buttons
            st.markdown("### Available Procedures")
            for sub in all_sub_processes:
                st.button(sub, key=f"btn_{sub}", on_click=select_procedure, args=(sub,))
