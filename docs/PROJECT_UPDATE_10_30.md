## TokenSmith: One-Page Project Update (Oct 30)

### Project Goal
My goal is to build an on-device RAG system that answers textbook questions (e.g., “What is atomicity?”, “How is OLAP different from OLTP?”) using local models via `llama.cpp`. I will optimize for accuracy and latency on a typical laptop, publish code on GitHub, and show measurable progress across iterations.

### Progress to Date (My Work)
- Week -2: I spent one week understanding the existing codebase and identifying low-effort, high-impact add-ons.
- Week -1: I implemented source citations in answers (feature 1) — almost done. Early checks indicate negligible latency overhead.

### Planned Incremental Features
1) Source citations in answers (status: nearly done)
   - What it is: attach source identifiers (section heading and/or filename with offsets) to each chunk used for an answer, and display citations at the end of the response.
   - Why it matters: improves traceability and user trust without impacting model behavior.
   - Implementation plan:
     - Add an option to the chat path to collect top-k chunk indices and map them to section headings/filenames.
     - Render a compact citations footer (e.g., “Sources: §14.1, §14.2”).
     - Add a toggle in config/CLI to enable/disable citations.

2) Latency instrumentation + JSONL logs
   - What it is: measure retrieve, rank, and generate durations; log per-query timings and configuration to `logs/*.jsonl`.
   - Why it matters: enables quantitative latency tracking (p50/p95) and before/after comparisons for changes.
   - Implementation plan:
     - Wrap retrieval, ranking, and generation with high-resolution timers.
     - Log per-query spans, top_k, pool_size, ensemble method, and model info.
     - Add a small summarizer script to compute per-stage p50/p95 and end-to-end stats.

3) Dynamic top_k to fit a token/char budget
   - What it is: adaptively choose `top_k` so that the composed prompt (system + chunks + question) stays under a fixed char/token budget.
   - Why it matters: reduces generator latency and avoids truncation while maintaining accuracy.
   - Implementation plan:
     - Estimate prompt size per chunk; sort chunks by rank; accumulate until budget is reached.
     - Fall back to a minimum `top_k` to guarantee context even for short budgets.
     - Expose budget and minimums via config/CLI; log realized `top_k` per query.

4) Micro-benchmark set and CLI
   - What it is: a lightweight, course-relevant set of 15–25 Q/A pairs derived from the textbook, plus a CLI to run evaluations and print terminal/HTML summaries.
   - Why it matters: provides comparable accuracy metrics across iterations, tied to the course content.
   - Implementation plan:
     - Curate question set (coverage: ACID, transactions, OLTP vs OLAP, indexing basics, B+ Trees).
     - Store as YAML or JSON; include expected text and optional keywords per question.
     - Add a run script/CLI that invokes the existing test harness and writes compact result summaries.

### Plan & Timeline
- Past 2 weeks:
  - Week -2: I spent one week understanding the existing codebase and pipeline.
  - Week -1: I implemented feature 1 (source citations) — almost done.
- Next 3 weeks (one feature per week before final presentation):
  - Week 1 (Nov 1–7): I will implement latency instrumentation + JSONL and run a small query set to get p50/p95 per stage.
  - Week 2 (Nov 8–14): I will implement dynamic top_k to a token/char budget and measure latency reduction vs accuracy.
  - Week 3 (Nov 15–19): I will build the micro-benchmark set + CLI and generate terminal/HTML/JSON outputs.
- Ongoing:
  - Finalize source citations in answers (feature 1; currently near completion).
  - Use semantic + keyword metrics for accuracy tracking (NLI optional depending on compute).
- Key dates:
  - Nov 18: Submit two-page final report with results and code links.
  - Nov 20: In-class presentation (if selected).

 

