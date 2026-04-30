# MI-RAG: Mutual Information Retrieval-Augmented Generation

A hybrid **Knowledge Graph + Retrieval-Augmented Generation (RAG)** pipeline for medical question answering.

MI-RAG combines:

* **Knowledge Graph retrieval (Neo4j)**
* **Semantic entity matching**
* **Mutual Information (MI)-based path scoring**
* **Self-pruning for hallucination reduction**
* **MapReduce-style answer synthesis using LLMs**
* **Dynamic Knowledge Graph updates**
* **Web fallback retrieval when the KG lacks coverage**

The system is designed to retrieve medically relevant knowledge paths, rank them intelligently, prune weak reasoning chains, and generate grounded answers with reduced hallucination risk.

---

## Architecture Overview

```text
User Question
      │
      ▼
Keyword Extraction
      │
      ▼
Semantic Entity Matching
      │
      ▼
Knowledge Graph Subgraph Retrieval
      │
      ▼
Mutual Information Path Scoring
      │
      ▼
Self-Pruning Mechanism
      │
      ▼
Triple Aggregation → Natural Language Context
      │
      ▼
MapReduce Answer Generation
      │
      ▼
Final Medical Answer
      │
      ▼
Knowledge Graph Update
```

If the retrieved subgraph is empty or insufficient, the pipeline automatically falls back to **DuckDuckGo-based web retrieval**.

---

# Features

## Knowledge Graph Driven Retrieval

* Uses **Neo4j** to store medical entities and relationships.
* Retrieves both **1-hop** and **2-hop reasoning paths**.
* Enables explainable retrieval instead of pure vector search.

## Semantic Entity Matching

* Uses `sentence-transformers` embeddings.
* Supports fuzzy semantic matching when exact entity matches fail.

## Mutual Information Path Scoring

* Scores retrieved paths using a probabilistic relevance estimation approach.
* Prioritizes reasoning chains that are highly informative for the query.

## Self-Pruning Mechanism

* Removes unstable or weak reasoning paths.
* Uses repeated LLM sampling + semantic similarity analysis.
* Helps reduce hallucinations and noisy evidence.

## MapReduce Answering

* Each retrieved fact generates a partial answer.
* Partial answers are merged into a final coherent response.
* Improves scalability for larger contexts.

## Dynamic Knowledge Graph Updates

* Extracts new triples from generated answers.
* Adds validated knowledge back into the graph.

## Web Retrieval Fallback

* Uses DuckDuckGo search when the KG lacks sufficient information.

---

# Tech Stack

| Component          | Technology            |
| ------------------ | --------------------- |
| Knowledge Graph    | Neo4j                 |
| Embeddings         | sentence-transformers |
| LLM Orchestration  | LangChain             |
| LLM Provider       | Groq API              |
| MI Scoring Model   | Qwen2-1.5B            |
| Backend Language   | Python                |
| Fallback Retrieval | DuckDuckGo Search     |

---

# Project Structure

```bash
MI-RAG/
│
├── aggregation/
│   └── aggregator.py
│
├── answering/
│   └── mapreduce_chain.py
│
├── evaluate/
│   └── evaluate.py
│
├── kg/
│   └── kg_construction.py
│
├── retrieval/
│   ├── entity_matcher.py
│   ├── keyword_extractor.py
│   └── subgraph_retriever.py
│
├── scoring/
│   ├── mi_scorer.py
│   └── pruner.py
│
├── update/
│   └── kg_updater.py
│
├── pipeline.py
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/MI-RAG.git
cd MI-RAG
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file in the root directory.

```env
GROQ_API_KEY=your_groq_api_key

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

---

# Neo4j Setup

## Install Neo4j

Download Neo4j from:

* [https://neo4j.com/download/](https://neo4j.com/download/)

Start the database locally and ensure the Bolt server is running.

Default connection:

```text
bolt://localhost:7687
```

---

# Knowledge Graph Construction

Populate the medical knowledge graph before running the pipeline.

```bash
python kg/kg_construction.py
```

This step:

* Extracts medical triples
* Validates relations
* Stores entities and relationships in Neo4j

---

# Running the Pipeline

Run:

```bash
python pipeline.py
```

Example questions:

```python
questions = [
    "What are the symptoms and treatments for diabetes?",
    "What medications are used for hypertension?",
    "What diagnostic tests are used for cancer?",
]
```

---

# Pipeline Walkthrough

## 1. Keyword Extraction

Extracts meaningful medical keywords from the user query.

---

## 2. Entity Matching

Maps extracted keywords to entities stored in the Knowledge Graph.

Supports:

* Exact matching
* Semantic similarity matching

---

## 3. Subgraph Retrieval

Retrieves:

* 1-hop relationships
* 2-hop reasoning chains

Example:

```text
Diabetes → symptoms_are → fatigue
Diabetes → common_medication_is → metformin
```

---

## 4. Mutual Information Scoring

Ranks paths based on how informative they are for the query.

This improves retrieval relevance compared to naive graph traversal.

---

## 5. Self-Pruning

Weak or unstable reasoning chains are removed.

The pruning stage:

* Samples multiple LLM outputs
* Measures semantic consistency
* Removes noisy evidence paths

---

## 6. Aggregation

Converts graph triples into readable natural language context.

---

## 7. MapReduce Answer Generation

### Map Step

Each fact produces a partial answer.

### Reduce Step

Partial answers are combined into one final response.

---

## 8. Knowledge Graph Update

New knowledge extracted from answers is added back into the graph.

---

# Example Output

```text
QUESTION:
What medications are used for hypertension?

FINAL ANSWER:
Common medications used for hypertension include ACE inhibitors,
beta blockers, calcium channel blockers, and diuretics.
```

---

# Supported Medical Relations

The system currently supports relations such as:

```text
can_treat_disease
may_suffer_from_disease
should_eat
common_medication_is
accompanied_by_symptoms_of
should_avoid_eating
symptoms_are
diagnostic_tests
```

---

# Research Concepts Used

This project integrates concepts from:

* Retrieval-Augmented Generation (RAG)
* Knowledge Graph Reasoning
* Mutual Information-based Retrieval
* Hallucination Reduction
* Semantic Similarity Matching
* Graph-based Explainability
* MapReduce-style LLM inference

---

# Future Improvements

* Hybrid vector + graph retrieval
* Medical ontology integration (UMLS / SNOMED CT)
* Multi-hop reasoning beyond 2 hops
* Confidence calibration
* Real-time medical knowledge ingestion
* Evaluation benchmark integration
* Streaming responses
* GPU acceleration for scoring models

---

# Limitations

* Depends heavily on Knowledge Graph quality.
* Current retrieval depth is limited to 2 hops.
* Medical responses should not replace professional clinical advice.
* LLM inference latency can increase with larger graphs.

---

# Disclaimer

This project is intended for research and educational purposes only.
It should not be used as a substitute for professional medical diagnosis or treatment.

---

# Contributors

Built as a research-oriented medical RAG pipeline focused on explainable retrieval and hallucination-aware answer generation.
