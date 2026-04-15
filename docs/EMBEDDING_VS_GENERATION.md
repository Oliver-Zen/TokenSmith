# Embedding Model vs Generation Model in TokenSmith

## Visual Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TOKENSMITH RAG PIPELINE                           │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
  PHASE 1: INDEXING (make run-index)
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT: Text Chunks                                                          │
│  Example: "B+ trees are balanced search trees used in databases..."         │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   EMBEDDING MODEL             │
        │   sentence-transformers/      │
        │   all-MiniLM-L6-v2            │
        │                               │
        │   • Purpose: Convert text     │
        │     → dense vectors           │
        │   • Input: Text string        │
        │   • Output: 384-dim vector   │
        │   • Used for: Search/Retrieval│
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   OUTPUT: Vector Embedding    │
        │   [0.23, -0.45, 0.12, ...]   │
        │   (384 numbers)               │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   FAISS Index                 │
        │   (Stores all embeddings)     │
        └──────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
  PHASE 2: QUERYING (chat mode)
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│  USER QUESTION: "What is a B+ tree?"                                        │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   EMBEDDING MODEL             │
        │   (Same as indexing phase)    │
        │                               │
        │   Question → Vector           │
        │   "What is a B+ tree?"        │
        │   → [0.15, -0.32, 0.08, ...]  │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   FAISS Similarity Search    │
        │   Find top-k similar chunks  │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   RETRIEVED CHUNKS           │
        │   "B+ trees are balanced..." │
        │   "A B+ tree has internal..."│
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   GENERATION MODEL            │
        │   qwen2.5-0.5b-instruct      │
        │   -q5_k_m.gguf                │
        │                               │
        │   • Purpose: Generate text   │
        │   • Input: Prompt with        │
        │     question + chunks         │
        │   • Output: Natural language │
        │     answer                    │
        │   • Used for: Answer creation│
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   OUTPUT: Generated Answer   │
        │   "A B+ tree is a balanced   │
        │    search tree structure..." │
        └──────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
  KEY DIFFERENCES
═══════════════════════════════════════════════════════════════════════════════

┌──────────────────────────────────┬──────────────────────────────────────────┐
│   EMBEDDING MODEL                │   GENERATION MODEL                       │
├──────────────────────────────────┼──────────────────────────────────────────┤
│                                  │                                          │
│  Model:                          │  Model:                                  │
│  all-MiniLM-L6-v2                │  qwen2.5-0.5b-instruct-q5_k_m.gguf      │
│                                  │                                          │
│  Purpose:                        │  Purpose:                                │
│  • Text → Vector (encoding)      │  • Text → Text (generation)              │
│  • Semantic similarity            │  • Answer generation                     │
│                                  │                                          │
│  Input:                          │  Input:                                  │
│  • Text string                   │  • Prompt with context + question        │
│                                  │                                          │
│  Output:                         │  Output:                                 │
│  • 384-dim vector (numbers)      │  • Natural language text                 │
│                                  │                                          │
│  When Used:                      │  When Used:                              │
│  • Indexing: chunks → vectors    │  • Chat: generate final answer           │
│  • Querying: question → vector   │                                          │
│                                  │                                          │
│  Library:                        │  Library:                                │
│  • sentence-transformers         │  • llama.cpp (via llama-cpp-python)     │
│                                  │                                          │
│  Size:                           │  Size:                                   │
│  • ~80MB (downloaded on first    │  • 498MB (local GGUF file)              │
│    use)                          │                                          │
│                                  │                                          │
│  Speed:                          │  Speed:                                  │
│  • Fast (batch processing)       │  • Slower (autoregressive generation)   │
│                                  │                                          │
│  Example:                        │  Example:                                │
│  Input:  "B+ tree structure"     │  Input:                                  │
│  Output: [0.23, -0.45, ...]      │    "Question: What is a B+ tree?        │
│                                  │     Context: B+ trees are..."            │
│                                  │  Output:                                 │
│                                  │    "A B+ tree is a balanced search      │
│                                  │     tree used in databases..."           │
│                                  │                                          │
└──────────────────────────────────┴──────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
  DATA FLOW SUMMARY
═══════════════════════════════════════════════════════════════════════════════

INDEXING:
  Documents → Chunks → [EMBEDDING MODEL] → Vectors → FAISS Index

QUERYING:
  Question → [EMBEDDING MODEL] → Query Vector → FAISS Search → Top Chunks
                                                                    ↓
  Answer ← [GENERATION MODEL] ← Prompt (Question + Chunks) ←────────┘


═══════════════════════════════════════════════════════════════════════════════
  ANALOGY
═══════════════════════════════════════════════════════════════════════════════

Think of it like a library system:

EMBEDDING MODEL = Library Catalog System
  • Converts book titles/descriptions → catalog numbers
  • Helps you FIND relevant books quickly
  • Doesn't read the books, just indexes them

GENERATION MODEL = Librarian
  • Reads the books you found
  • Synthesizes information
  • Gives you a comprehensive answer

