## Final Project Report

[https://github.com/Oliver-Zen/TokenSmith/commits/main/]()

### Project Overview
TokenSmith is an on-device RAG (Retrieval-Augmented Generation) system that answers database textbook questions using local models via `llama.cpp`. The system combines FAISS vector search and BM25 lexical retrieval with ensemble ranking to provide accurate, traceable answers on typical laptop hardware. All development focused on improving traceability, measurability, user experience, and performance without compromising accuracy.

### Completed Features
Over the past three weeks, I implemented five incremental features with measurable impact on system functionality and user experience:

**1. Source Citations **
- **What it does**: Attaches section identifiers (e.g., "Sources: §14.1, §14.2") to answers, showing which textbook sections informed the response.
- **Why it matters**: Improves answer traceability and user trust without modifying model behavior or adding latency overhead.
- **Implementation**: Extended metadata pipeline to capture section headings during indexing, created format_citations() to extract section numbers from top-k chunks, and added `--citations` CLI flag and config toggle (default: false).
- **Result**: Citations display cleanly after answers with zero measurable latency impact. ~105 lines across 5 files.

**2. Latency Instrumentation with JSONL Logging**
- **What it does**: Measures retrieval, ranking, and generation stage durations using high-resolution timers; logs per-query timings with configuration metadata to `logs/*.jsonl`.
- **Why it matters**: Enables quantitative latency analysis (p50/p95/p99), supports before/after comparisons for optimizations, and provides data-driven performance insights.
- **Implementation**: Wrapped pipeline stages with `time.perf_counter()`, extended RunLogger with log_latency() method and custom JSON serializer for numpy types, created summarize_latency.py script to compute percentile statistics across sessions.
- **Result**: Per-stage timing data logged to JSONL format, analyzer script provides p50/p95/p99 breakdowns. ~346 lines across 8 files including 268-line analysis tool.

**3. Streaming Output**
- **What it does**: Displays LLM responses token-by-token in real-time instead of waiting for complete generation.
- **Why it matters**: Significantly improves perceived responsiveness and user experience, especially for longer answers (typical 5-15 second generation time).
- **Implementation**: Created run_llama_cpp_streaming() with state machine parser (BEFORE_ANSWER → IN_ANSWER → AFTER_ANSWER) to extract content from llama-cli stdout, modified chat loop to handle both streaming and non-streaming modes based on `--stream` flag.
- **Result**: Interactive streaming display with no loss of functionality (citations still appear, latency still tracked). ~183 lines across 5 files.

**4. Parallel Retrieval**
- **What it does**: Runs FAISS and BM25 retrievers concurrently using ThreadPoolExecutor instead of sequentially.
- **Why it matters**: Reduces retrieval stage latency by ~40-50% by exploiting retriever independence and multi-core parallelism.
- **Implementation**: Wrapped retriever.get_scores() calls in executor.submit(), collected results via as_completed(), preserved original sequential fallback for single-retriever setups. Added `--parallel-retrieval` flag (default: true).
- **Result**: Typical retrieval time reduced from ~200ms to ~110ms for hybrid retrieval with negligible code complexity. ~37 lines across 4 files.

**5. Query Caching with LRU Eviction**
- **What it does**: Caches complete answers for repeated queries using an LRU (Least Recently Used) cache with configurable size (default: 128 entries).
- **Why it matters**: Provides instant responses (<1ms) for repeated questions, improving interactive exploration and benchmark reruns.
- **Implementation**: Created QueryCache class with normalized keys (lowercase + strip for better hit rate), implemented manual LRU eviction, integrated cache check before pipeline execution and storage after generation (non-streaming only). Added `--query-cache` flag and cache_size config.
- **Result**: Cache hits bypass entire pipeline with instant response. Useful for demos, testing, and interactive sessions. ~102 lines across 4 files.


### Technical Architecture
The system pipeline consists of four stages with optional features controlled via config and CLI flags:

1. **Retrieval**: FAISS (vector similarity) + BM25 (lexical matching) retrieve top pool_size chunks
   - Optional: Parallel execution (default: enabled)
   - Optional: Query cache check (default: disabled)
2. **Ranking**: Ensemble ranker (RRF/weighted/linear) combines scores, selects top_k
3. **Generation**: llama.cpp runs local GGUF model to produce answer
   - Optional: Streaming output (default: disabled)
   - Optional: Latency timing (default: disabled)
4. **Post-processing**: Citation formatting and result caching
   - Optional: Source citations (default: disabled)
   - Optional: Cache storage (default: disabled)

All features are optional and backward-compatible; existing tests and workflows continue unchanged with defaults.

### Challenges and Solutions

**Challenge 1: Model Output Issues**
- **Problem**: After implementing citations, LLM returned empty responses.
- **Root cause**: Duplicate answer() function definitions in generator.py (second overwriting first) + incorrect model path in config.
- **Solution**: Removed duplicate function, corrected model path to build/llama.cpp/models/, created debug script to diagnose generation issues.

**Challenge 2: JSON Serialization Errors**
- **Problem**: Logging failed with "Object of type int64 is not JSON serializable" when writing chunk indices.
- **Root cause**: FAISS and BM25 return numpy int64 types which aren't JSON-serializable by default.
- **Solution**: Added custom _json_serializer() to handle numpy integer, float, and array types.

**Challenge 3: Streaming Parser Complexity**
- **Problem**: Extracting answer text from llama-cli stdout with prompt templates and special tokens.
- **Root cause**: llama-cli outputs full prompt + answer markers + generation.
- **Solution**: Implemented state machine parser (BEFORE_ANSWER → IN_ANSWER → AFTER_ANSWER) with marker detection for robust extraction.