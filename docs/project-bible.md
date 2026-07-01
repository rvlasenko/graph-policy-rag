# graph-policy-rag Project Bible

# Project Scope

Build a production-style GraphRAG platform for marketplace policy retrieval.
The goal is to learn architecture, not to build a demo.

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

---

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

### Decision 6

FileSystemDocumentLoader is responsible for loading documents from the file system.

**Why**

The loader knows where documents come from, but not how they are used later.

---

## Learnings

- `tmp_path` creates an isolated temporary directory for each test.
- Unit tests should verify behavior through the public API, not private implementation details.

---

## Questions

- Why is `Document` a domain entity?
- Why shouldn't `Chunk` contain the whole `Document`?
- Why is hashing the document content a poor choice for the document `id`?

---

# Principles

- Form a hypothesis before looking up the answer.
- Verify tooling behavior using the official documentation.
- Prefer team conventions when multiple solutions are correct.
- Evolve models instead of trying to design perfect ones from the beginning.
- Avoid generic `utils.py`; start with local private helpers and extract them only when reuse appears.

---

# Learnings

- `uv init` and `uv init .` both work.
- A contract is an agreement between system components about exchanged data.
- Domain entities represent business concepts rather than implementation details.

---

# Vocabulary

## Adapter

A component that translates external input into the application's internal format and back.

## Contract

An agreement between system components describing the structure of exchanged data.

## Domain Entity

An object that represents a business concept and is independent of frameworks, storage and transport.