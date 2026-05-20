import os
from pathlib import Path

import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Text-only Gemini embedding model documented by Google.
EMBEDDING_MODEL = "models/gemini-embedding-001"

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "travel_knowledge"


def chunk(text, size=300, overlap=50):
    """Split text into word-based chunks with a configurable overlap."""
    if size <= 0:
        raise ValueError("size must be greater than 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    words = text.split()
    chunks = []
    step = size - overlap

    for start in range(0, len(words), step):
        part = words[start:start + size]
        if part:
            chunks.append(" ".join(part))

    return chunks


def _embed(text, task_type):
    """Return one Gemini embedding vector."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type=task_type,
    )
    return result["embedding"]


def _destination_key(destination):
    """Match a destination to its lowercase knowledge filename."""
    return destination.strip().lower().replace(" ", "_")


def knowledge_file_exists(destination):
    """Check whether knowledge/<destination>.txt exists."""
    key = _destination_key(destination)
    return (KNOWLEDGE_DIR / f"{key}.txt").is_file()


def build_index(chunk_size=300, overlap=50):
    """
    Read all destination text files, embed their chunks, and save them in ChromaDB.
    Returns the number of chunks indexed.
    """
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Rebuild from source files so changed text/settings do not leave stale chunks.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    all_ids = []
    all_texts = []
    all_embeddings = []
    all_metadatas = []

    for file_path in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        destination = file_path.stem.lower()
        text = file_path.read_text(encoding="utf-8").strip()

        for number, text_chunk in enumerate(chunk(text, chunk_size, overlap)):
            all_ids.append(f"{destination}_{number}")
            all_texts.append(text_chunk)
            all_embeddings.append(_embed(text_chunk, "retrieval_document"))
            all_metadatas.append({"source": destination})

    if all_ids:
        collection.add(
            ids=all_ids,
            documents=all_texts,
            embeddings=all_embeddings,
            metadatas=all_metadatas,
        )

    return len(all_ids)


def retrieve(destination, k=5):
    """
    Retrieve the top-k chunks for one destination.
    Returns (chunk_texts, source_names).
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not knowledge_file_exists(destination):
        return [], []

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    destination_key = _destination_key(destination)
    query = f"Travel information about {destination.strip()}: attractions, areas, food, and local tips."
    query_embedding = _embed(query, "retrieval_query")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"source": destination_key},
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    chunks = documents
    sources = [metadata["source"] for metadata in metadatas]
    return chunks, sources
