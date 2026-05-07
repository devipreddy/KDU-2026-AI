# Hybrid GraphRAG for Multi-Hop Reasoning

## Overview

This repository contains a complete end-to-end implementation of a production-inspired Hybrid GraphRAG system designed to demonstrate:

* limitations of traditional vector-based RAG systems,
* graph-based multi-hop reasoning,
* hybrid retrieval orchestration,
* Neo4j knowledge graph traversal,
* semantic + graph retrieval fusion,
* and LangGraph-based workflow orchestration.

The project is implemented across three progressive phases:

1. Phase 1 — Traditional Vector RAG Failure Analysis
2. Phase 2 — Knowledge Graph Construction
3. Phase 3 — Production-Grade Hybrid GraphRAG

The implementation combines:

* ChromaDB
* Neo4j AuraDB
* LangGraph
* LangChain
* Groq-hosted LLMs
* Hybrid retrieval orchestration

to build a topology-aware multi-hop reasoning pipeline.

-
# Problem Statement

Traditional Retrieval-Augmented Generation (RAG) systems rely heavily on semantic vector similarity.

While effective for:

* semantic search,
* summarization,
* and contextual QA,

they struggle significantly with:

* relationship-heavy reasoning,
* ownership chains,
* dependency traversal,
* and multi-hop inference.

This occurs because:

* chunking fragments relationships,
* embeddings preserve semantics but not topology,
* and vector retrieval cannot deterministically traverse entity relationships.

This repository demonstrates how GraphRAG architectures solve these limitations by integrating:

* knowledge graphs,
* graph traversal,
* semantic retrieval,
* and orchestration workflows.

# Repository Structure

```text
.
├── Phase-1-Vector-RAG-Blindspot.ipynb
├── Phase-2-Knowledge-Graph.ipynb
├── Phase-3-Hybrid-GraphRAG-LangGraph.ipynb
├── requirements.txt
├── .env.example
└── README.md
```


# Architecture Overview

## Final Hybrid GraphRAG Architecture

```text
User Query
    │
    ▼
LLM-Based Query Router
    │
    ├── VECTOR Retrieval
    │       └── ChromaDB Semantic Search
    │
    ├── GRAPH Retrieval
    │       └── Neo4j Cypher Traversal
    │
    └── HYBRID Retrieval
            ├── Semantic Retrieval
            └── Graph Traversal

            ▼

Context Fusion Layer
    │
    ▼
Final LLM Reasoning
    │
    ▼
Multi-Hop Answer
```

# Phase 1 — Multi-Hop Blindspot in Traditional Vector RAG

## Objective

Demonstrate why traditional vector-based RAG systems fail at multi-hop reasoning.

## Implementations

* Token-aware document chunking
* ChromaDB semantic retrieval
* RetrievalQA pipeline
* Multi-hop retrieval attack queries
* Retrieval diagnostics
* Similarity score inspection
* Chunk fragmentation analysis

## Key Demonstration

The notebook demonstrates how semantic retrieval fails to reconstruct ownership chains spread across multiple disconnected chunks.

Example failure scenario:

```text
Acme Holdings Ltd.
    ↓
Shell Alpha LLC
    ↓
Beta Investments Inc.
    ↓
Delta Trading Corp.
    ↓
John Smith
```

Traditional vector retrieval retrieves semantically similar chunks but fails to preserve explicit relationship topology.

# Phase 2 — Knowledge Graph Construction

## Objective

Convert unstructured enterprise text into a structured knowledge graph.

## Implementations

* LLM-based triple extraction
* JSON triple parsing
* Entity normalization
* Predicate canonicalization
* Neo4j graph ingestion
* MERGE-based deduplication
* Graph traversal queries
* Ownership chain reconstruction
* Knowledge graph visualization

## Knowledge Graph Features

The graph preserves:

* ownership relationships,
* executive dependencies,
* shell company structures,
* and multi-hop organizational topology.

## Technologies Used

* Neo4j AuraDB
* Cypher
* NetworkX
* LangChain
* Groq LLMs


# Phase 3 — Production-Grade Hybrid GraphRAG

## Objective

Build a production-inspired Hybrid GraphRAG architecture combining:

* vector retrieval,
* graph traversal,
* orchestration workflows,
* and LLM reasoning.

## Core Features

### Query Router

An LLM-based router dynamically determines whether a query requires:

* VECTOR retrieval
* GRAPH traversal
* HYBRID retrieval


### Cypher Generation

User questions are converted into Cypher queries using:

* schema-aware prompting,
* ontology grounding,
* and constrained graph schemas.


### Cypher Validation Layer

Implemented safeguards include:

* forbidden operation filtering,
* unsafe query prevention,
* and schema validation.

### Retry and Self-Correction

If Cypher execution fails:

* Neo4j errors are captured,
* the LLM regenerates corrected Cypher,
* and execution retries automatically.


### Hybrid Retrieval

The system combines:

* semantic retrieval from ChromaDB,
* deterministic graph traversal from Neo4j,
* and LLM-based context fusion.

### LangGraph Orchestration

LangGraph is used for:

* conditional routing,
* workflow orchestration,
* retry handling,
* retrieval coordination,
* and state management.

# Technologies Used

| Component              | Technology        |
| ---------------------- | ----------------- |
| Vector Database        | ChromaDB          |
| Graph Database         | Neo4j AuraDB      |
| Workflow Orchestration | LangGraph         |
| LLM Framework          | LangChain         |
| LLM Provider           | Groq              |
| Embeddings             | OpenAI Embeddings |
| Graph Query Language   | Cypher            |


## Configure Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key

GROQ_API_KEY=your_groq_key

NEO4J_URI=neo4j+s://<instance>.databases.neo4j.io

NEO4J_USERNAME=neo4j

NEO4J_PASSWORD=your_password
```


# Running the Project


Each phase builds upon the previous phase.

# Key Concepts Demonstrated

* Multi-hop reasoning
* Vector retrieval limitations
* Knowledge graph construction
* Entity resolution
* Predicate normalization
* Neo4j traversal
* Cypher generation
* Retry-based query correction
* Hybrid retrieval orchestration
* LangGraph workflows
* Context fusion
* Deterministic relationship traversal

# Example Query Types

## Vector Query

```text
What is Delta Trading Corp under investigation for?
```

Uses:

* semantic retrieval


## Graph Query

```text
Which company indirectly owns Delta Trading Corp?
```

Uses:

* graph traversal


## Hybrid Query

```text
Explain how Acme Holdings controls the company where John Smith works.
```

Uses:

* vector retrieval
* graph traversal
* context fusion


# Why Hybrid GraphRAG?

| Vector Retrieval           | Graph Retrieval                  |
| -------------------------- | -------------------------------- |
| Semantic understanding     | Exact relationships              |
| Contextual similarity      | Deterministic traversal          |
| Flexible language matching | Multi-hop reasoning              |
| Weak topology awareness    | Strong relationship preservation |

Hybrid GraphRAG combines both approaches to achieve:

* semantic richness,
* explicit relationship reasoning,
* and explainable multi-hop inference.

# Real-World Applications

This architecture is highly applicable to:

* Financial fraud detection
* Enterprise intelligence
* Supply chain analysis
* Cybersecurity investigations
* Legal discovery
* Biomedical knowledge systems
* Organizational dependency analysis
* Risk and compliance systems

# Final Outcome

This repository demonstrates the transition from:

* semantic-only retrieval systems

to

* production-inspired Hybrid GraphRAG architectures capable of:

  * deterministic graph traversal,
  * topology-aware reasoning,
  * semantic enrichment,
  * and explainable multi-hop inference.

The final system integrates:

* vector retrieval,
* graph traversal,
* orchestration workflows,
* and LLM reasoning

into a complete Hybrid GraphRAG pipeline.
