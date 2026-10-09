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

# Day 5 — Vector Store

## Decisions

### Decision 17

`VectorStore` stores embeddings but does not own chunk text.

**Why**

Vector search and chunk storage have different responsibilities. Search results can be resolved to domain chunks through a separate repository.

---

### Decision 18

Vector search returns `chunk_id` and similarity `score`, but not the stored vector.

**Why**

The caller needs the chunk identity and relevance score. Returning the full vector would expose unnecessary storage details.

---

### Decision 19

`VectorStore` uses `upsert` instead of append-only `add`.

**Why**

Repeated ingestion and document updates must replace existing embeddings instead of creating duplicate search results for the same chunk.

---

## Decision 20

`VectorStore` has an explicit vector dimension.

**Why**

All vectors stored in one index must have the same dimension. Explicitly configuring the dimension allows incompatible embeddings to be rejected during `upsert` instead of failing later during search.

---

## Decision 21

`InMemoryVectorStore.upsert` validates the entire batch before modifying stored embeddings.

**Why**

Batch upsert should be atomic from the caller's perspective. If any embedding has an incompatible model, wrong vector dimension or zero vector, the whole operation is rejected and the store remains unchanged.

---

## Decision 22

`VectorStore` rejects zero vectors.

**Why**

Cosine similarity is undefined for a zero vector because its norm is zero. Rejecting zero vectors at the store boundary prevents invalid data from being persisted and avoids delayed failures during search.

---

## Decision 23

`VectorStore.search` validates the query embedding before calculating similarity.

The query must:

* use the same embedding model as the store;
* have the configured vector dimension;
* not be a zero vector.

**Why**

Input validation should happen at the public method boundary. This produces earlier and clearer errors instead of relying on failures inside the cosine similarity implementation.

## Learnings

* A vector index has a fixed embedding model and vector dimension.
* Embeddings created by different models or with different dimensions cannot be compared meaningfully.
* A zero vector has zero norm, so cosine similarity cannot be calculated.
* Batch validation must finish before any writes occur to preserve atomicity.
* A component should reject data that would leave it in a state it cannot process correctly later.
* Validation at a public method boundary gives clearer errors than accidental failures inside lower-level calculations.

## Questions

* Should validation of `NaN` and infinite vector values happen in `ChunkEmbedding`, the embedding adapter or `VectorStore`?
* Should vector validation remain inside `InMemoryVectorStore` or later become shared logic for all vector store adapters?
* Should invalid vector data use dedicated exception types instead of `ValueError`?

---

# Day 6 — Query Pipeline

## Decisions

### Decision 24

`QueryRequest` is an internal application contract.

**Why**

HTTP, CLI and other adapters should convert external input into the same application-level request. This keeps query execution independent from the transport layer.

---

### Decision 25

`Retriever` accepts query text and returns application-level `RetrievalHit` objects.

**Why**

The retriever hides retrieval implementation details such as query embedding, vector search, keyword search, graph expansion and reranking.

`QueryPipeline` should depend on the retrieval capability rather than directly coordinating a specific vector store.

---

### Decision 26

`RetrievalHit` is separate from `VectorSearchResult`.

**Why**

`VectorSearchResult` is the contract of the vector store, while `RetrievalHit` is the application retrieval contract.

This allows the retrieval implementation to evolve from vector-only search to hybrid or graph-based retrieval without changing the query pipeline.

## Learnings

- A retriever converts a user query into relevant knowledge candidates.
- A vector store is one possible implementation detail inside a retriever.
- Application contracts should not expose infrastructure-specific result types.
- Ports describe capabilities required by the application.
- Adapters implement those capabilities using concrete technologies.

## Questions

- Should `Retriever` return `Sequence[RetrievalHit]` or `list[RetrievalHit]`?
- Which component should resolve `chunk_id` into chunk text and document metadata?
- Should query embedding happen inside the retriever or in a separate query-embedding port?

---

### Decision 27

Chunk content is stored separately from vector embeddings.

**Why**

The vector store is responsible for similarity search and returns chunk identifiers with scores.

A separate `ChunkRepository` resolves chunk identifiers into domain `Chunk` objects. This prevents the retrieval application from depending on vector-store-specific payload storage.

### Decision 28

`ChunkRepository.get` returns `Chunk | None`.

**Why**

Vector and chunk storage can become temporarily inconsistent. The caller should explicitly decide how to handle a missing chunk instead of receiving an accidental low-level lookup error.

---

### Decision 29

Chunk storage is represented by the `ChunkRepository` port.

The repository supports both storing chunks and resolving a chunk by identifier.

**Why**

Vector storage contains embeddings used for similarity search, while chunk storage contains the original domain objects required for context assembly and citations.

Both stores are updated during ingestion.

---

### Decision 30

Re-ingesting a document removes its stale chunks from both the chunk repository and the vector store.

**Why**

Chunk ids are derived from the document id and the chunk position. When a document becomes shorter, `upsert` replaces the remaining positions but leaves the old trailing chunks in place, and those keep appearing in search results. Ingestion compares the chunks currently stored for the document with the newly produced ones and deletes the difference.

New data is written before stale data is deleted, so a failed delete leaves extra results rather than missing ones. The two stores are not updated transactionally; the context assembler tolerates a hit whose chunk is missing (Decision 28).

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