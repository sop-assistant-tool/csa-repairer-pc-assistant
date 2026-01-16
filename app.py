import os
import streamlit as st
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# --- 1. SIMPLE PASSWORD AUTHENTICATION ---
# This replaces the complex Google/Microsoft login for easier setup
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.set_page_config(page_title="Login | PartsCheck", page_icon="🔒")
    st.markdown("# 🔒 PartsCheck Support Hub")
    
    password = st.text_input("Enter Password to Access", type="password")
    if st.button("Log In"):
        # It checks Streamlit Secrets for APP_PASSWORD, defaults to 'PartsCheck2026'
        if password == st.secrets.get("APP_PASSWORD", "PartsCheck2026"):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# --- 2. CONFIG & DATA ---
# Ensure these folders and files exist in your GitHub repository
CSV_PATH = "processed_docs/final_rag_dataset.csv"
IMG_DIR = "processed_docs/images"

@st.cache_resource
def load_data():
    if not os.path.exists(CSV_PATH):
        st.error(f"CSV file not found at: {CSV_PATH}. Please check your GitHub folder structure.")
        st.stop()
    
    df = pd.read_csv(CSV_PATH, encoding='utf-8').dropna(subset=['process', 'sub_process'])
    unique_subs = df[['process', 'sub_process']].drop_duplicates()
    
    search_docs = [Document(page_content=f"{r['process']} {r['sub_process']}",
                   metadata={'sub': r['sub_process']}) for _, r in unique_subs.iterrows()]
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(search_docs, embeddings)
    return vectorstore, df, unique_subs['sub_process'].tolist()

vectorstore, df, all_sub_processes = load_data()

# Initialize Groq LLM (Make sure GROQ_API_KEY is in Streamlit Secrets)
llm = ChatGroq(
    model="llama-3.2-3b-preview", 
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
# page_config must be called only once
st.set_page_config(page_title="PartsCheck Assistant", layout="wide")

with st.sidebar:
    st.markdown("👤 **User:** PartsCheck Member")
    
    # Placeholder for logo - Update the URL once you have your logo hosted
    # st.image("https://raw.githubusercontent.com/YOUR_USER/YOUR_REPO/main/logo.png", use_container_width=True)
    
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
display_container = st.container()

with display_container:
    # Welcome message for new sessions
    if not st.session_state.chat_history and not st.session_state.target_sub:
        with st.container(border=True):
            st.markdown("""
            ### Welcome to the PartsCheck Support Hub
            I am your dedicated **SaaS Support Assistant**. I help navigate the platform and optimize your quoting workflow.
            
            **How to use:**
            * **Search SOPs:** Ask *"How do I create a quote?"*
            * **General Help:** Ask about platform features or industry standards.
            """)
            st.info("💡 **Try asking:** 'How do I integrate with iBodyShop?'")

    # Display the specific SOP Guide if one is selected
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
        
        if st.button("❌ Close Guide & Return to Chat"):
            st.session_state.target_sub = None
            st.rerun()
            
    # Display Chat History
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

# --- 6. INTELLIGENT ROUTING & CHAT ---
if question := st.chat_input("Ask about an SOP or the industry..."):
    st.session_state.target_sub = None
    
    with st.chat_message("user"):
        st.write(question)
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.status("🔍 Analyzing request...", expanded=True) as status:
            # LLM-based Router to decide between SOP or CHAT
            router_prompt = f"""
            Available SOPs: {', '.join(all_sub_processes)}
            Input: "{question}"
            Task: If this is a request for a specific procedure, output ONLY '[SOP] Name'. 
            If it is a general question, output ONLY '[CHAT]'.
            """
            route_response = llm.invoke(router_prompt).content.strip()
            status.update(label="Intent Identified!", state="complete", expanded=False)

        if "[SOP]" in route_response:
            # Uses semantic search to find the closest SOP match
            matches = vectorstore.similarity_search(question, k=1)
            st.session_state.target_sub = matches[0].metadata['sub']
            st.rerun()
        else:
            # Professional SaaS Expert Chat Response
            chat_prompt = f"""
            You are the PartsCheck SaaS Expert. 
            Tone: Professional, Corporate, subtly Australian.
            Constraint: Focus only on parts procurement, quoting, and workflow.
            Question: {question}
            """
            response = llm.invoke(chat_prompt).content
            st.write(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
