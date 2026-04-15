## TokenSmith – High-Level Overview

### Purpose
TokenSmith is a Retrieval-Augmented Generation (RAG) application that lets you index textbook content and answer questions locally using `llama.cpp` for inference and FAISS/BM25 for retrieval.

### Architecture (Core Modules)
- **Configuration (`src/config.py`)**: `QueryPlanConfig` holds chunking, retrieval/ranking, generation, and testing settings. Loads from `config/config.yaml`.
- **Preprocessing**:
  - `src/preprocessing/extraction.py`: Splits `data/book_without_image.md` into sections based on headings (e.g., `## 14.1 ...`).
  - `src/preprocessing/chunking.py`: Recursively chunks section text via LangChain’s `RecursiveCharacterTextSplitter`. Preserves `<table>...</table>` blocks when `keep_tables=True`.
- **Indexing (`src/index_builder.py`)**: Chunks sections, embeds with a local GGUF embedding model via `src/embedder.py` (wrapper around `llama_cpp`), builds FAISS and BM25, and writes artifacts under `index/<strategy>/`:
  - `{prefix}.faiss`, `{prefix}_bm25.pkl`, `{prefix}_chunks.pkl`, `{prefix}_sources.pkl`, `{prefix}_meta.pkl`.
- **Retrieval (`src/retriever.py`)**: 
  - `FAISSRetriever` (semantic) and `BM25Retriever` (lexical) produce scored candidates.
  - `load_artifacts()` loads indexes/chunks/sources.
  - `apply_seg_filter()` optional segment filtering before top-k.
- **Ranking (`src/ranking/ranker.py`)**: `EnsembleRanker` fuses retriever signals using weighted Linear or RRF. Re-ranking via cross-encoder exists in `src/ranking/reranker.py` (currently disabled in main flow).
- **Planning (`src/planning/*.py`)**: `HeuristicQueryPlanner` classifies each query as definitional, explanatory, analytical, or procedural using query length, question type, comparison terms, and entity density, then emits a per-query `RAGConfig`.
- **Generation (`src/generator.py`)**: Builds a prompt (modes: baseline, tutor, concise, detailed), shells out to `llama-cli`, and extracts the answer between markers.
- **Entrypoint (`src/main.py`)**:
  - `index` mode builds artifacts.
  - `chat` mode loads artifacts once, retrieves + ranks chunks per query, then generates.

### Data Flow
1) Markdown → sections (extraction)
2) Sections → chunks (chunking strategy)
3) Chunks → embeddings (local GGUF via `llama_cpp`)
4) Embeddings → FAISS index; Chunks → BM25 index
5) Query → planner → per-query retrievers/ranker settings → fused ranking → top-k chunks
6) Top-k chunks + system prompt → `llama-cli` → answer

### Artifacts & Paths
- Artifacts directory: `index/<strategy>/` where `<strategy>` is derived from chunk config (e.g., `sections`).
- Default prefix: `textbook_index` (overridable via CLI `--index_prefix`).
- Markdown source: `data/book_without_image.md`.

### Configuration Highlights (`config/config.yaml`)
- `chunk_mode`: currently supports section-based recursive strategy via `SectionRecursiveConfig`.
- Retrieval: `top_k`, `pool_size`, `ensemble_method` (rrf/linear), `ranker_weights`, `rrf_k`.
- Planning: `planner_mode` (`none` or `heuristic`) controls whether TokenSmith uses static retrieval settings or chooses a per-query execution plan.
- Generation: `model_path`, `max_gen_tokens`, `system_prompt_mode` (baseline/tutor/concise/detailed).
- Testing toggles: `disable_chunks`, `use_golden_chunks`, `output_mode`, `metrics`.

### Models
- Embeddings: a local GGUF embedding model path is supplied via `embed_model` and loaded by `src/embedder.py` using `llama_cpp` with `embedding=True`.
- Generator: default example in README is `models/qwen2.5-0.5b-instruct-q5_k_m.gguf`.

### CLI & Make Targets
- `python -m src.main index` or `make run-index` to build artifacts.
- `python -m src.main chat` (interactive) or `make run-chat`.
- `make build` sets up conda env, builds/detects `llama.cpp`, installs the package.

### Testing (`tests/`)
- `tests/test_benchmarks.py` runs benchmark questions through the same pipeline.
- Benchmark results now record latency and planner-selected query category for quality/latency comparisons.
- Pluggable metrics in `tests/metrics/*` (semantic similarity, keyword match, NLI). Outputs terminal or HTML reports and JSON logs.

### Notes
- `src/generator.py` resolves `llama-cli` via `$LLAMA_CPP_BINARY`, `src/llama_path.txt`, or PATH.
- Ensure embedding model dimension matches FAISS index dimension; both are created during indexing.
