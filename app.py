import os
import streamlit as st
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

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
    search_docs = [
        Document(
            page_content=f"{r['process']} {r['sub_process']}",
            metadata={'sub': r['sub_process']}
        ) for _, r in unique_subs.iterrows()
    ]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(search_docs, embeddings)
    return vectorstore, df, unique_subs['sub_process'].tolist()

vectorstore, df, all_sub_processes = load_data()

# LLM for routing/general chat only (optional fallback)
llm = ChatOllama(model="llama3.1:8b", temperature=0)  # Fast routing; change to 3.2 if needed

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
    # Logo (use raw string for Windows paths)
    logo_path = r"C:\Users\ErickMortera\Pictures\PartsCheck logo.png"
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

    # SOP Guide Display
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

    # Chat History Display (non-SOP mode)
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

# --- FIXED INPUT & ROUTING ---
if question := st.chat_input("Ask about an SOP or the industry..."):
    # Always clear the SOP guide when a new question is typed
    st.session_state.target_sub = None
    
    with st.chat_message("user"):
        st.write(question)
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        # Use st.status for visible thinking
        with st.status("🔍 Analyzing request...", expanded=True) as status:
            router_prompt = f"""
            Available SOPs: {', '.join(all_sub_processes)}
            
            Task: Classify this user input: "{question}"
            - If it matches an SOP exactly (e.g. 'how do i credit in partscheck'), output: [SOP] Name
            - If it's a general question or greeting, output: [CHAT]
            - If it's a 'How to' that needs disambiguation, output: [DISAMBIGUATE]
            
            Output ONLY the label.
            """
            route_response = llm.invoke(router_prompt).content.strip()
            status.update(label="Intent Identified!", state="complete", expanded=False)

        # ACTION LOGIC
        if "[SOP]" in route_response:
            matched_name = route_response.replace("[SOP]", "").strip()
            # Find closest match if the AI didn't copy perfectly
            closest_match = None
            for sub in all_sub_processes:
                if sub.lower() in matched_name.lower() or matched_name.lower() in sub.lower():
                    closest_match = sub
                    break
            if closest_match:
                st.session_state.target_sub = closest_match
                st.rerun()
            else:
                # Fallback to search if exact match string fails
                route_response = "[DISAMBIGUATE]"

        if "[CHAT]" in route_response:
            chat_prompt = f"""
            You are the PartsCheck Smart Assistant, a professional support expert for the PartsCheck SaaS platform.
            
            OFFICIAL PRODUCT SCOPE (ONLY talk about these):
            - Parts Procurement: Efficiently sourcing and ordering parts from suppliers.
            - Quoting: Creating and managing parts quotes for smash repairers.
            - Supplier Integration: Connecting repairers with a network of automotive parts suppliers.
            - Workflow Optimization: Streamlining the administrative side of parts management.
            
            GUIDELINES:
            1. TONE: Professional, efficient, and courteous. Avoid excessive slang.
            2. IDENTITY: You are a technical product expert. You help businesses optimize their workflows using PartsCheck.
            3. AUSTRALIAN CONTEXT: Use a subtle Australian greeting (e.g., "G'day" or "Hello") but maintain a corporate SaaS standard.
            4. RESTRICTION: Do not mention physical repairs, "crumpled cars," or acting as a mechanic. Focus entirely on software, procurement, and supplier-repairer efficiency.
            
            STRICT RULES:
            - DO NOT mention inventory management, physical car repairs, or financial accounting unless specifically asked how PartsCheck integrates with those external systems.
            - If a user asks for a function we don't have, say: "Currently, PartsCheck focuses on procurement and quoting. I can show you how to optimize those areas, or discuss our integrations."
            - Keep the tone professional, corporate, and subtly Australian.
            
            User Question: {question}
            """
            
            with st.spinner("Consulting Product Specs..."):
                resp = llm.invoke(chat_prompt).content
                
            st.session_state.chat_history.append({"role": "assistant", "content": resp})
            st.rerun()
            
        elif "[DISAMBIGUATE]" in route_response or "[SOP]" not in route_response:
            st.markdown("### 📖 Which procedure do you need?")
            matches = vectorstore.similarity_search(question, k=4)
            cols = st.columns(2)
            for i, doc in enumerate(matches):
                name = doc.metadata['sub']
                cols[i % 2].button(f"👉 {name}", key=f"btn_{i}", on_click=select_procedure, args=(name,))
