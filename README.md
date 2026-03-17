# RAG Dataset Project

This project is a Retrieval-Augmented Generation (RAG) pipeline that answers questions using information from datasets (e.g., yearly data for three companies).

## ✅ What it does
- Loads structured data from local files (CSV/JSON/TXT)
- Converts the data into text "documents"
- Splits documents into chunks and creates embeddings
- Stores embeddings in a Chroma vector store
- Uses a text generation model to answer user questions using retrieved context

## 🚀 Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Put your datasets inside `data/`.
   - Supported formats: `.pdf`, `.csv`, `.json`, `.txt`
   - Suggested naming: `company_a.pdf`, `company_b.pdf`, `company_c.pdf` (or `.csv`, etc.)

3. (Optional but recommended) Create a `.env` file with a Hugging Face token to use the Inference API (avoids local model issues):
   ```env
   HUGGINGFACE_API_TOKEN=hf_...
   ```
   Get your token from [Hugging Face](https://huggingface.co/settings/tokens).

4. Run the pipeline:
   ```bash
   python rag_pipeline.py
   ```

## 🔧 Notes
- If you do not provide an HF token, the script will try to use local models, but may fail due to NumPy compatibility issues on some systems.
- The script caches embeddings in `./chroma_db/` so reruns are fast.
- The pipeline loads all files in `data/`, processes them, and allows interactive questions.
