import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

SYSTEM_PROMPT = """You are an assistant that answers questions using only the information found in the provided context.
The context contains yearly financial data for three companies.

When answering:
- Use only the provided context (the retrieved dataset content).
- Do not invent any facts or speculate beyond what is in the context.
- If the answer is not in the context, respond exactly:

  "I don’t have that information in the provided context."

- If the answer is present, provide it clearly and, when helpful, begin with:

  "Based on the provided context..."
"""

PROMPT_TEMPLATE = """{system_prompt}

Context:
{context}

Question:
{question}

Answer:"""


def load_documents(data_dir: str):
    """Load PDF documents from the data directory."""
    from langchain_core.documents import Document
    from pypdf import PdfReader

    documents = []

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for file_name in sorted(os.listdir(data_dir)):
        file_path = os.path.join(data_dir, file_name)
        if os.path.isdir(file_path):
            continue

        ext = file_name.lower().rsplit(".", 1)[-1]

        if ext == "pdf":
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            documents.append(Document(page_content=text, metadata={"source": file_path}))
        else:
            continue

    return documents


def get_llm():
    """Return an LLM for text generation."""
    try:
        from transformers import pipeline
        from langchain_huggingface import HuggingFacePipeline
        text_pipe = pipeline("text-generation", model="microsoft/DialoGPT-small", device_map="auto")
        return HuggingFacePipeline(pipeline=text_pipe)
    except ImportError:
        raise RuntimeError("Transformers is not available.")
        


def simple_split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Simple text splitter to avoid langchain dependencies."""
    from langchain_core.documents import Document
    texts = []
    for doc in documents:
        content = doc.page_content
        start = 0
        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            texts.append(Document(page_content=chunk, metadata=doc.metadata))
            start += chunk_size - chunk_overlap
            if start >= len(content):
                break
    return texts


def main():
    # Load and preprocess documents.
    documents = load_documents(DATA_DIR)
    if not documents:
        raise RuntimeError(f"No supported documents found in {DATA_DIR}. Put your PDF/CSV/JSON/TXT files there.")

    texts = simple_split_documents(documents, chunk_size=500, chunk_overlap=100)

    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    from langchain_chroma import Chroma
    # Persist embeddings for faster re-runs.
    if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    else:
        db = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIR)

    retriever = db.as_retriever(search_kwargs={"k": 2})

    from langchain_core.prompts import PromptTemplate
    prompt = PromptTemplate(
        input_variables=["system_prompt", "context", "question"],
        template=PROMPT_TEMPLATE,
    )

    llm = get_llm()

    # Simple RAG chain using retriever and LLM
    while True:
        query = input("Enter your question (or 'quit' to exit): ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            break

        # Retrieve relevant documents
        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs])

        # Build the prompt
        full_prompt = PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            context=context,
            question=query
        )

        # Get LLM answer
        answer = llm.invoke(full_prompt)

        print("\n=== Answer ===")
        print(answer)

        if docs:
            print("\n=== Retrieved sources ===")
            for i, d in enumerate(docs, start=1):
                print(f"--- Source {i} ---")
                print(d.page_content[:500].strip())
                print()



if __name__ == "__main__":
    main()
