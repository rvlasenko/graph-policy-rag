# Graph Policy RAG

Retrieval service for marketplace policy documents (returns, shipping, seller onboarding), built step by step toward a production-style GraphRAG platform.

> **Status: work in progress.** Today the service does vector retrieval with citations. Graph-based retrieval and LLM answer generation are planned; see [Roadmap](#roadmap).

## What works today

- Policy documents are split into overlapping word-based chunks.
- Chunks are embedded locally with Sentence Transformers (`all-MiniLM-L6-v2`), so no paid API is needed.
- Embeddings live in an in-memory vector store that searches by cosine similarity.
- `POST /query` returns the most relevant chunks, each with its score and a citation (`chunk_id`, `document_id`, `source_uri`).
- Re-ingesting a document is idempotent: changed chunks are replaced, and chunks that no longer exist are removed from both stores.

## Architecture

The code follows a ports-and-adapters layout. Application logic depends on protocols (ports), not on FastAPI, Sentence Transformers or a specific store.

```
                 ┌──────────────── ingestion ────────────────┐
 Document ──▶ TextChunker ──▶ ChunkEmbedder ──▶ ChunkRepository + VectorStore

                 ┌──────────────── query ────────────────────┐
 POST /query ──▶ QueryPipeline ──▶ Retriever ──▶ ContextAssembler ──▶ QueryResponse
                                   (embed query,   (resolve chunk ids,
                                    vector search)  dedupe, size budget)
```

| Layer | Package | Responsibility |
|---|---|---|
| Domain | `domain/` | `Document`, `Chunk`, `ChunkEmbedding`: framework-free data models |
| Application | `application/` | Use cases (`QueryPipeline`, `IngestionService`), ports (`Retriever`, `ChunkRepository`, `ContextAssembler`) and internal contracts |
| Adapters | `api/`, `embeddings/`, `vector_store/`, `chunk_repository/`, `retrieval/`, `ingestion/` | FastAPI, Sentence Transformers, in-memory storage, file-system loader |
| Wiring | `bootstrap.py` | Builds the object graph in one place |

Key design choices:

- **Identity is separate from version.** A document `id` comes from its path and its `version` from a content hash. The same holds for chunks, which makes stale embeddings detectable.
- **Vectors are stored separately from chunk text.** The vector store returns only `chunk_id` and `score`; `ChunkRepository` resolves ids into chunks. Either store can be swapped (Qdrant, pgvector, Postgres) independently.
- **Validation at boundaries.** The vector store rejects vectors from another model, with the wrong dimension, or equal to zero, and it validates a whole batch before writing.
- **The transport is an adapter.** `QueryPipeline` works the same from HTTP, CLI or tests.

All 30 design decisions, each with its rationale, are recorded in [`docs/project-bible.md`](docs/project-bible.md).

## Quick start

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run uvicorn graph_policy_rag.api.app:app --reload
```

On first start the embedding model (~90 MB) is downloaded from Hugging Face. At startup the API ingests a built-in demo returns policy.

```bash
curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "How many days do I have to return a product?", "top_k": 3}'
```

Example response (shortened):

```json
{
  "context": {
    "items": [
      {
        "chunk": {
          "document_id": "returns-policy",
          "text": "Customers may return unused products within 30 days of delivery. ...",
          "source_uri": "demo://returns-policy"
        },
        "score": 0.76
      }
    ]
  },
  "citations": [
    {"chunk_id": "4426…", "document_id": "returns-policy", "source_uri": "demo://returns-policy"}
  ]
}
```

Interactive API docs are served at http://127.0.0.1:8000/docs.

## Tests

```bash
uv run pytest
```

The tests use fake embedding models, so they run in well under a second and do not download anything.

## Known limitations

- All storage is in memory, so the index is rebuilt on every start.
- The API serves only the built-in demo document. The file-system loader exists but is not wired to startup yet.
- There is no LLM step: the response is retrieved context with citations, not a generated answer.
- No graph yet: retrieval is purely vector-based.
- Ingestion is not transactional across the two stores. If the second write fails, chunk and vector storage can disagree until the document is ingested again. The context assembler skips hits whose chunk is missing.

## Roadmap

- [x] Document loading, chunking, local embeddings
- [x] Vector store, retriever, context assembly with citations
- [x] FastAPI query endpoint
- [ ] Ingest policy documents from a directory at startup
- [ ] LLM answer generation grounded in the retrieved context
- [ ] Knowledge graph of policy entities and relations; graph-expanded retrieval
- [ ] Persistent storage (vector DB + chunk store), Docker Compose
- [ ] Retrieval evaluation set (recall@k)
- [ ] CI: tests, ruff, mypy
