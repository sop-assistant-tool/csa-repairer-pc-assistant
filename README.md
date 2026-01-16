# PartsCheck SOP Assistant

This is an internal tool designed to guide customer repair advisors and smash repairers through PartsCheck Standard Operating Procedures (SOPs).  
It provides step-by-step instructions with screenshots for common tasks like receipting, quoting, onboarding, crediting, and reporting.

## Features
- Search for procedures by keyword (e.g., "how do i credit in partscheck", "add supplier")
- Full step-by-step guidance with images shown inline
- No hallucinations — displays only official SOP content

## How to Use
- Type your question in the chat input
- The tool will show the matching SOP procedure with all steps and relevant screenshots

## Data Source
- SOP data is stored in `processed_docs/final_rag_dataset.csv`
- Images are in `images/`

Built with Streamlit for fast, Python-based deployment.