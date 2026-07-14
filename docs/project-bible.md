# Project Scope

Build a production-style GraphRAG platform for marketplace policy retrieval.

---

# Day 1 — Architecture Boundaries

## Decisions

### Decision 1

FastAPI is used for the online query API, not for document ingestion.

**Why**

Documents are processed by the ingestion pipeline.

---

### Decision 2

`QueryRequest` and `QueryResponse` are internal application contracts.

FastAPI and CLI are adapters that convert external input into these contracts.

**Why**

This keeps the application independent from the transport layer.

---

### Decision 3

FastAPI depends on `QueryPipeline`, not vice versa.

**Why**

`QueryPipeline` contains application logic and should work with HTTP, CLI, Slack or tests.

---

### Decision 4

Use `uv` for project and dependency management.

**Why**

`uv` provides reproducible environments, dependency management and lockfiles.

## Learnings

- `uv init` and `uv init .` both work.
- A contract is an agreement between system components about exchanged data.

## Questions

- Why should FastAPI depend on QueryPipeline instead of the opposite?
- Why are QueryRequest and QueryResponse internal contracts?
- What is the difference between document ingestion and the online query API?

---

# Day 2 — Document Ingestion

## Decisions

### Decision 5

Document identity and document version are different concepts.

- `id` is generated from the relative document path.
- `version` is generated from the document content hash.

**Why**

A document should keep the same identity when its content changes, while the version should change whenever the content changes.

---

### Decision 6

FileSystemDocumentLoader is responsible for loading documents from the file system.

**Why**

The loader knows where documents come from, but not how they are used later.

## Learnings

- Unit tests should verify behavior through the public API, not private implementation details.
- `tmp_path` creates an isolated temporary directory for each test.
- Domain entities represent business concepts rather than implementation details.

## Questions

- Why is `Document` a domain entity?
- Why is hashing the document content a poor choice for the document `id`?

---

# Day 3 — Chunking

## Decisions

### Decision 7

`Chunk` stores `document_id`, not the full `Document`.

**Why**

A chunk is a retrieval unit, but it must remain linked to its source document without duplicating the whole document.

---

### Decision 8

`TextChunker` lives in `chunking`, not in `domain` or `ingestion`.

**Why**

Chunking transforms `Document` domain objects into `Chunk` domain objects and is independent of the document source.

---

### Decision 9

Initial chunking is word-based using `str.split()`.

**Why**

It keeps the first implementation simple and dependency-free. Token-based chunking can be introduced later when model-specific limits become important.

---

### Decision 10

Shared hashing logic lives in `common/hashing.py`.

**Why**

Hashing is reused by multiple components. Extracting it into a focused module avoids duplication without introducing a generic `utils.py`.

### Decision 11

`TextChunker` keeps the final partial chunk.

**Why**

Policy documents may contain important information at the end. Losing content is worse than producing a shorter chunk.

## Learnings

- An invariant is a condition that must always hold true for a model or algorithm to work correctly.
- `step = chunk_size_words - overlap_words`.
- Overlap preserves context between neighboring chunks.

## Questions

- Why can't we embed the whole document as one vector?
- Why does `Chunk` need `index`?
- Is `source_uri` a property of `Chunk` or a denormalized copy from `Document`?

---

# Day 4 — Embeddings

## Decisions

### Decision 12

`Chunk` has a version generated from its text.

**Why**

Chunk identity represents its position within a document, while its version changes when the text at that position changes. This allows stale embeddings to be detected.

---

### Decision 13

`ChunkEmbedding` is a separate domain model.

**Why**

An embedding has its own lifecycle and metadata and should not be stored directly inside `Chunk`.

---

### Decision 14

`ChunkEmbedding` stores `chunk_id`, `chunk_version`, `vector`, and `model`.

**Why**

This links the vector to its source chunk, allows stale embeddings to be detected, and identifies the embedding space in which the vector was created.

---

### Decision 15

Embedding processes chunks in batches and preserves their order.

**Why**

Embedding models can process multiple texts more efficiently than repeated single-item calls. Preserving order keeps each result aligned with its source chunk.

---

### Decision 16

Sentence Transformers is the first embedding model implementation.

**Why**

It provides local batch inference without coupling the application to a paid external API. The implementation adapts the library output to the internal `EmbeddingModel` contract.

## Learnings

- Embeddings created by different models belong to different vector spaces and should not be compared directly.
- `frozen=True` does not make a nested `list` immutable.
- `tuple[float, ...]` represents an immutable sequence containing any number of floats.
- Chunk identity and chunk content version are different concepts.
- Batch processing reduces the number of calls to an embedding model.
- A smoke test verifies that a real integration starts and produces a minimally valid result.
- Sentence Transformers returns NumPy vectors, which are converted at the adapter boundary into plain Python floats.

## Questions

- Should domain models validate their own invariants using `__post_init__`?
- Which embedding implementation should be used first?
- Where should compatibility of vector dimensions be validated?

---

# Principles

- Form a hypothesis before looking up the answer.
- Verify tooling behavior using the official documentation.
- Prefer team conventions when multiple solutions are correct.
- Evolve models instead of trying to design perfect ones from the beginning.
- Avoid generic `utils.py`; start with local private helpers and extract them only when reuse appears.

---

# Vocabulary

## Adapter

A component that translates external input into the application's internal format and back.

## Contract

An agreement between system components describing the structure of exchanged data.

## Domain Entity

An object that represents a business concept and is independent of frameworks, storage and transport.

## Domain

Core concepts from the problem space.

## Application

Use cases and workflows that coordinate domain objects.

## Infrastructure

Technical details such as frameworks, databases, file systems, APIs and external services.