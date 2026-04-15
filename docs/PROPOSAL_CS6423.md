# Cost-Based Query Optimization for RAG Pipelines

**Course:** CS 6423 — Database System Implementation (Spring 2026)
**Student:** Oliver Xie
**Repository:** [TokenSmith](https://github.com/georgia-tech-db/TokenSmith)

## Problem

TokenSmith is a local-first Retrieval-Augmented Generation (RAG) system that answers questions over textbooks using local LLMs. Its pipeline has many configurable knobs — retrieval mode (FAISS, BM25, hybrid), ranking strategy (RRF vs. linear fusion), retriever weights, re-ranking (cross-encoder on/off), query enhancement (HyDE, query expansion, decomposition), top-k, and candidate pool size — but currently uses a single static configuration for every query. This one-size-fits-all approach is suboptimal: a simple definitional query ("What is a foreign key?") wastes latency on expensive stages like HyDE and cross-encoder re-ranking, while a complex analytical query ("Contrast OLTP and data analytics goals") may need those stages to retrieve sufficient context. The system lacks the ability to adapt its execution strategy to the query at hand.

## Importance

This problem mirrors a core challenge in database systems: query optimization. Just as a DBMS enumerates candidate execution plans and selects one based on cost estimates, a RAG system can treat its pipeline stages as operators and choose an execution plan per query. This is especially important for local-first systems where compute is constrained — unnecessary LLM calls (HyDE, double prompting) can double latency with no quality benefit for simple queries. A cost-based optimizer can deliver the right quality-latency tradeoff per query, making local RAG practical for interactive use.

## Approach

The work is organized into three tiers of goals:

**75% Goals — Rule-Based Query Planner Integration**
- Integrate the existing (but unused) `QueryPlanner` into the main pipeline so that each query gets a per-query `RAGConfig`.
- Extend the heuristic classifier beyond simple keyword matching: use query length, presence of comparison terms, question type (what/why/how), and entity density to classify queries into categories (definitional, explanatory, analytical, procedural).
- Map each category to a hand-tuned pipeline configuration (e.g., definitional → BM25-heavy, no HyDE, no re-ranking; analytical → hybrid retrieval, HyDE enabled, cross-encoder re-ranking).
- Evaluate on the benchmark suite: compare per-category static configs against the one-size-fits-all baseline on both latency and answer quality.

**100% Goals — Cost-Based Plan Enumeration and Selection**
- Define the plan space: each plan is a combination of (retrieval mode, ranker weights, top-k, HyDE on/off, re-rank on/off, double-prompt on/off). Enumerate valid plans.
- Build a cost model from historical query logs. Each pipeline stage has a latency cost (measured from instrumentation) and an estimated quality contribution (measured from benchmark evaluations). Use simple regression or lookup tables calibrated on the benchmark suite.
- Implement a plan selector that, given a query classification and a latency budget, enumerates candidate plans, estimates total cost and expected quality, and selects the plan on the Pareto frontier.
- Collect runtime statistics (per-stage latency, retrieval score distributions) and persist them to update the cost model across sessions.

**125% Goals — Feedback-Driven Adaptive Optimization**
- Implement a lightweight quality estimator: after generation, use an LLM-as-judge (or retrieval-score heuristics) to score the answer quality without requiring ground truth.
- Close the feedback loop: use per-query quality signals to update cost model weights online via exponential moving averages. Plans that underperform get penalized; plans that overperform get rewarded.
- Implement plan caching: cache (query-class, plan) mappings so that similar queries reuse previously successful plans without re-running the optimizer.
- Multi-objective analysis: produce Pareto frontier visualizations showing quality vs. latency tradeoffs across different plan strategies and query categories.

## Validation

- **Correctness:** Unit tests verify that the planner produces valid `RAGConfig` objects for each query class, that the cost model returns consistent estimates, and that the feedback loop converges (weights stabilize over repeated queries).
- **Integration:** End-to-end tests confirm that the optimizer is invoked on every chat query and that the selected plan is actually used by the pipeline (verified via log inspection).

## Evaluation

- **Benchmark suite:** Evaluate on the existing 11-question benchmark (and extend to 25+ questions covering all query categories). Metrics: semantic similarity, BLEU, keyword recall against expected answers.
- **Latency:** Measure end-to-end latency per query with and without the optimizer. Report per-stage latency breakdowns.
- **Baselines:** Compare four configurations: (1) static default config, (2) rule-based planner (75% goal), (3) cost-based optimizer (100% goal), (4) feedback-adaptive optimizer (125% goal).
- **Quality vs. latency tradeoff:** Plot Pareto frontiers for each approach. The optimizer should achieve comparable quality to the best static config at lower average latency, or higher quality at comparable latency.
- **Adaptation:** Show that the feedback loop improves plan selection over time by plotting quality and latency across successive queries.

## Resources

- **Hardware:** Apple Silicon Mac (Metal acceleration for local LLM inference).
- **Models:** Qwen3-Embedding-4B (GGUF, embedding), Qwen2.5-3B-Instruct (GGUF, generation) — already in the repository.
- **Data:** Database System Concepts (Silberschatz et al.) textbook, already indexed with FAISS + BM25.
- **Software:** Python, llama-cpp-python, FAISS, sentence-transformers (cross-encoder), scikit-learn (regression for cost model), matplotlib (Pareto frontier plots).
- **Workload:** Extended benchmark suite (25+ questions across query categories) derived from the existing 11-question suite.
