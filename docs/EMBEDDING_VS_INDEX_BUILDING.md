# Embedding vs Index-Building in TokenSmith

## Key Distinction

**Embedding** = Converting text → vectors (ONE step)
**Index-Building** = Complete process to create searchable structures (MULTIPLE steps)

---

## Visual Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INDEX-BUILDING (Complete Process)                        │
│                    Called by: make run-index                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: EXTRACTION & CHUNKING                                              │
│  ─────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  Input: data/book_without_image.md                                          │
│    ↓                                                                         │
│  extract_sections_from_markdown()                                           │
│    → Sections (by ## headings)                                              │
│    ↓                                                                         │
│  DocumentChunker.chunk()                                                    │
│    → Text chunks (recursive splitting)                                       │
│                                                                              │
│  Output: List of text chunks                                                │
│  ["B+ trees are balanced...", "A B+ tree has...", ...]                     │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: EMBEDDING ⭐ (This is "Embedding")                                  │
│  ─────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  Input: Text chunks                                                          │
│  ["B+ trees are balanced...", "A B+ tree has...", ...]                     │
│    ↓                                                                         │
│  SentenceTransformer.encode()                                                │
│  (Uses: sentence-transformers/all-MiniLM-L6-v2)                              │
│    ↓                                                                         │
│  Output: NumPy array of vectors                                              │
│  embeddings = [                                                              │
│    [0.23, -0.45, 0.12, ..., 0.89],  # Vector for chunk 0                   │
│    [0.15, -0.32, 0.08, ..., 0.76],  # Vector for chunk 1                   │
│    ...                                                                       │
│  ]  # Shape: (N_chunks, 384)                                                │
│                                                                              │
│  ⚠️  EMBEDDING STOPS HERE - Just vectors in memory                          │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: BUILD FAISS INDEX                                                  │
│  ─────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  Input: Embeddings (vectors)                                                │
│    ↓                                                                         │
│  faiss.IndexFlatL2(dim=384)                                                 │
│  index.add(embeddings)                                                      │
│    ↓                                                                         │
│  faiss.write_index(index, "textbook_index.faiss")                           │
│                                                                              │
│  Output: Searchable FAISS index file                                        │
│  (Enables fast similarity search)                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: BUILD BM25 INDEX                                                   │
│  ─────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  Input: Text chunks (original text, not vectors)                             │
│    ↓                                                                         │
│  preprocess_for_bm25() → tokenized chunks                                    │
│  BM25Okapi(tokenized_chunks)                                                │
│    ↓                                                                         │
│  pickle.dump(bm25_index, "textbook_index_bm25.pkl")                         │
│                                                                              │
│  Output: Searchable BM25 index file                                         │
│  (Enables keyword-based search)                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: SAVE ARTIFACTS                                                     │
│  ─────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  Save to disk:                                                               │
│  • textbook_index_chunks.pkl  (original text chunks)                        │
│  • textbook_index_sources.pkl (source file paths)                           │
│  • textbook_index_meta.pkl    (metadata: section, chunk_id, etc.)          │
│                                                                              │
│  Output: Complete index artifacts ready for retrieval                       │
└─────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
  DETAILED BREAKDOWN
═══════════════════════════════════════════════════════════════════════════════

┌──────────────────────────────────┬──────────────────────────────────────────┐
│   EMBEDDING                      │   INDEX-BUILDING                        │
├──────────────────────────────────┼──────────────────────────────────────────┤
│                                  │                                          │
│  What it is:                     │  What it is:                            │
│  • ONE step in the pipeline      │  • COMPLETE pipeline (5 steps)           │
│  • Text → Vector conversion       │  • Creates searchable structures        │
│                                  │                                          │
│  Code location:                  │  Code location:                          │
│  src/index_builder.py            │  src/index_builder.py                    │
│  Lines 93-98                     │  Lines 40-128 (entire function)          │
│                                  │                                          │
│  Input:                          │  Input:                                  │
│  • List of text chunks           │  • Markdown file                         │
│  ["chunk1", "chunk2", ...]       │  • Configuration                        │
│                                  │                                          │
│  Process:                        │  Process:                                │
│  embedder.encode(chunks)          │  1. Extract sections                    │
│                                  │  2. Chunk sections                       │
│                                  │  3. Embed chunks ⭐                     │
│                                  │  4. Build FAISS index                    │
│                                  │  5. Build BM25 index                     │
│                                  │  6. Save artifacts                       │
│                                  │                                          │
│  Output:                         │  Output:                                 │
│  • NumPy array of vectors        │  • FAISS index file                      │
│  • Shape: (N, 384)               │  • BM25 index file                       │
│  • In memory only                │  • Chunks pickle file                    │
│                                  │  • Sources pickle file                   │
│                                  │  • Metadata pickle file                  │
│                                  │  • All saved to disk                    │
│                                  │                                          │
│  Purpose:                        │  Purpose:                                │
│  • Convert text to numbers       │  • Create searchable database            │
│  • Enable semantic similarity     │  • Enable fast retrieval                │
│                                  │  • Persist for later use                │
│                                  │                                          │
│  When it happens:                │  When it happens:                       │
│  • During index-building         │  • When you run: make run-index         │
│  • Also during querying          │  • One-time setup                       │
│    (for user questions)          │                                          │
│                                  │                                          │
│  Example:                        │  Example:                                │
│  Input:                          │  Input:                                  │
│    "B+ trees are balanced..."    │    data/book_without_image.md            │
│                                  │                                          │
│  Output:                         │  Output:                                 │
│    [0.23, -0.45, 0.12, ...]      │    index/sections/                      │
│                                  │      ├── textbook_index.faiss           │
│                                  │      ├── textbook_index_bm25.pkl         │
│                                  │      ├── textbook_index_chunks.pkl       │
│                                  │      ├── textbook_index_sources.pkl     │
│                                  │      └── textbook_index_meta.pkl         │
│                                  │                                          │
└──────────────────────────────────┴──────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
  ANALOGY
═══════════════════════════════════════════════════════════════════════════════

Think of building a library:

EMBEDDING = Converting book titles to catalog numbers
  • Takes: "Database Systems" → "DB-001"
  • Just the conversion step
  • Doesn't create the catalog system

INDEX-BUILDING = Building the entire library system
  • Organizing books on shelves (chunking)
  • Creating catalog numbers (embedding) ⭐
  • Building the card catalog (FAISS index)
  • Building the keyword index (BM25 index)
  • Saving all records (artifacts)
  • Complete, searchable library ready to use


═══════════════════════════════════════════════════════════════════════════════
  CODE REFERENCE
═══════════════════════════════════════════════════════════════════════════════

# EMBEDDING (Step 2 only)
# src/index_builder.py, lines 93-98

print(f"Embedding {len(all_chunks):,} chunks...")
embedder = SentenceTransformer(embedding_model_path)
embeddings = embedder.encode(
    all_chunks, batch_size=4, show_progress_bar=True
)
# embeddings is a NumPy array: (N_chunks, 384)


# INDEX-BUILDING (Complete function)
# src/index_builder.py, lines 40-128

def build_index(...):
    # Step 1: Extract and chunk
    sections = extract_sections_from_markdown(...)
    for section in sections:
        chunks = chunker.chunk(section['content'])
        all_chunks.extend(chunks)
    
    # Step 2: Embed (this is "embedding")
    embedder = SentenceTransformer(...)
    embeddings = embedder.encode(all_chunks, ...)
    
    # Step 3: Build FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, ...)
    
    # Step 4: Build BM25 index
    bm25_index = BM25Okapi(tokenized_chunks)
    pickle.dump(bm25_index, ...)
    
    # Step 5: Save artifacts
    pickle.dump(all_chunks, ...)
    pickle.dump(sources, ...)
    pickle.dump(metadata, ...)


═══════════════════════════════════════════════════════════════════════════════
  SUMMARY
═══════════════════════════════════════════════════════════════════════════════

• EMBEDDING is Step 2 of index-building
  → Converts text chunks to vector embeddings
  → Uses embedding model (sentence-transformers/all-MiniLM-L6-v2)
  → Output: NumPy array in memory

• INDEX-BUILDING is the complete 5-step process
  → Includes embedding as one step
  → Creates searchable FAISS and BM25 indexes
  → Saves all artifacts to disk
  → Makes documents searchable for queries

Relationship: Embedding ⊂ Index-Building

