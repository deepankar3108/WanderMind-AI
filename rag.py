
import os
from pathlib import Path

import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key
load_dotenv(override=True)

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=api_key)

# Gemini embedding model
EMBEDDING_MODEL = "models/gemini-embedding-001"

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "travel_knowledge"


def chunk(text, size=300, overlap=50):
    """Split text into word-based chunks."""
    if size <= 0:
        raise ValueError("size must be greater than 0")

    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be smaller than size")

    words = text.split()
    chunks = []
    step = size - overlap

    for start in range(0, len(words), step):
        part = words[start:start + size]
        if part:
            chunks.append(" ".join(part))

    return chunks


def _embed(text, task_type):
    """Generate a Gemini embedding."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type=task_type,
    )
    return result["embedding"]


def _destination_key(destination):
    return destination.strip().lower().replace(" ", "_")


def knowledge_file_exists(destination):
    """Check whether a destination knowledge file exists."""
    key = _destination_key(destination)
    return (KNOWLEDGE_DIR / f"{key}.txt").is_file()


def build_index(chunk_size=300, overlap=50):
    """Build the ChromaDB knowledge index."""
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

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
        file_text = file_path.read_text(encoding="utf-8").strip()

        for number, text_chunk in enumerate(
            chunk(file_text, chunk_size, overlap)
        ):
            all_ids.append(f"{destination}_{number}")
            all_texts.append(text_chunk)

            all_embeddings.append(
                _embed(text_chunk, "retrieval_document")
            )

            all_metadatas.append({"source": destination})

    if all_ids:
        collection.add(
            ids=all_ids,
            documents=all_texts,
            embeddings=all_embeddings,
            metadatas=all_metadatas,
        )

    return len(all_ids)


def _get_collection():
    """Return the existing knowledge collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


def retrieve(destination, k=5):
    """Retrieve chunks for a particular destination."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not knowledge_file_exists(destination):
        return [], []

    collection = _get_collection()

    query = (
        f"Travel information about {destination.strip()}: "
        "attractions, areas, food, and local tips."
    )

    query_embedding = _embed(query, "retrieval_query")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"source": _destination_key(destination)},
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    chunks = documents
    sources = [
        metadata["source"] for metadata in metadatas
    ]

    return chunks, sources


def retrieve_query(question, k=5):
    """Search across all knowledge files using a general question."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not KNOWLEDGE_DIR.exists():
        return [], []

    if not list(KNOWLEDGE_DIR.glob("*.txt")):
        return [], []

    collection = _get_collection()

    if collection.count() == 0:
        return [], []

    query_embedding = _embed(question, "retrieval_query")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()),
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    chunks = documents
    sources = [
        metadata["source"] for metadata in metadatas
    ]

    return chunks, sources
