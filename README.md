# WanderMind AI — RAG setup

## What changed

- `rag.py` splits destination text into word chunks, embeds them with Gemini, and stores the vectors in persistent ChromaDB.
- `knowledge/<destination>.txt` is the source of truth. The filename (lowercase, spaces replaced by underscores) becomes the `source` metadata.
- `app.py` has a **Use RAG** sidebar toggle, chunk-size/overlap settings, a retrieval `k` setting, and a button to rebuild the index.
- When RAG is enabled but the destination file or index is missing, the app displays a notice and uses the original prompt without retrieved context.
- Hotel phone numbers and addresses are not requested or displayed because generated contact details are not verified.
- Weather and currency code, and the existing page styling and itinerary layout, are retained.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root:

   ```env
   GOOGLE_API_KEY=your_gemini_api_key
   ```

3. Add real, verified text to one file per destination in `knowledge/`.
   Use lowercase filenames, for example `goa.txt`, `jaipur.txt`, or `new_delhi.txt`.
   The included `goa.txt` is only a placeholder and contains no destination facts.

4. Build (or rebuild) the persistent vector index:

   ```bash
   python -c "from rag import build_index; print(build_index(chunk_size=300, overlap=50))"
   ```

   The index is saved in `chroma_db/`. Re-run this command whenever you change a knowledge file
   or want to compare chunk-size/overlap settings. You can also use the app's sidebar button.

5. Run Streamlit:

   ```bash
   streamlit run app.py
   ```

## Experiment settings

- Chunk size and overlap are configurable in `build_index(chunk_size=..., overlap=...)`.
- Retrieval count is configurable in `retrieve(destination, k=...)`.
- In the app, change the chunk-size/overlap sliders and click **Rebuild knowledge index**.
  Change the `Retrieved chunks (k)` slider to compare retrieval breadth.

## Embedding model

This project uses `models/gemini-embedding-001` with `retrieval_document` and
`retrieval_query`, as these task types are supported for Gemini Embedding 1.
Google's current documentation identifies `gemini-embedding-2` as the latest multimodal
model, but says `gemini-embedding-001` remains available for text-only use cases and
supports the `task_type` field used here. Check Google's embeddings documentation before
changing model names or migrating the embedding API:
https://ai.google.dev/gemini-api/docs/embeddings

If you change embedding models, rebuild the entire ChromaDB index. Vectors from different
embedding models should not be mixed.
