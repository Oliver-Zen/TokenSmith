[https://github.com/Oliver-Zen/TokenSmith](https://github.com/Oliver-Zen/TokenSmith)

## Project Overview

My project studies how database-style query optimization ideas can improve a local Retrieval-Augmented Generation system. The base system is TokenSmith, an on-device RAG pipeline that indexes a database textbook, retrieves relevant chunks with FAISS and BM25, optionally re-ranks them, and generates answers with local GGUF models through `llama.cpp`. The specific goal of my project is to move TokenSmith away from a one-size-fits-all pipeline and toward adaptive execution planning. Instead of always using the same retrieval weights, candidate pool size, and expensive enhancement steps for every question, I want the system to choose a query plan based on the type of question being asked.

The motivation comes directly from database systems. In a DBMS, the optimizer does not execute every SQL query with the same plan; it chooses an execution strategy based on the query structure and cost estimates. I am applying the same idea to RAG. For example, a short definitional question such as “What is conflict serializability?” does not need the same processing budget as a comparative or analytical question such as “How do strict two-phase locking and basic two-phase locking differ in their effect on recoverability?” A local-first system especially benefits from this kind of adaptation because latency and compute are limited on laptop hardware. For this checkpoint, my focus has been the 75% goal from the proposal: integrating a rule-based planner into the actual TokenSmith pipeline and making it observable through benchmarks and logs. All experiments for this checkpoint use the course-provided Silberschatz textbook and the default retrieval setting of `top_k = 10`.

## Current Progress

The main completed milestone is that TokenSmith now has a functioning per-query planning layer rather than only a static configuration. I implemented a `HeuristicQueryPlanner` and connected it to the real chat and API paths, so each query can receive its own effective `RAGConfig` at runtime. The planner is no longer dead code sitting off to the side. It is now part of the execution path used by the CLI chat loop, the FastAPI server, and the benchmark harness.

The planner currently classifies queries into four categories: definitional, explanatory, analytical, and procedural. The classification logic goes beyond the original keyword-only draft. It now looks at multiple query features, including question type (`what`, `why`, `how`, `compare`), query length, comparison-oriented language, procedural cues, and entity density. That matters because many student questions in the textbook domain are phrased differently even when they belong to the same conceptual type. A simple keyword match was too brittle, while the updated heuristic layer gives me a stronger baseline for the midterm checkpoint.

Each query class now maps to a hand-tuned pipeline configuration. Definitional questions bias heavily toward BM25 and disable expensive stages such as HyDE and cross-encoder reranking. Explanatory questions use a more balanced hybrid setup but still avoid unnecessary enhancement. Analytical questions use a larger candidate pool, hybrid retrieval, HyDE, and cross-encoder reranking, because these questions usually require broader evidence and better ranking quality. Procedural questions use a wider candidate set and reranking to help capture multi-step explanations that may be distributed across several chunks. This mapping is the core of the rule-based optimization milestone described in the proposal.

I also refactored the pipeline so these per-query settings are actually consumed correctly. Earlier in the semester, TokenSmith built retrievers and rankers once at startup and then reused a single static configuration. That design was incompatible with query-specific plans. I changed the runtime to keep the expensive index artifacts loaded once while rebuilding only the lightweight per-query retrieval and ranking configuration. This preserves efficiency while allowing each query to use different weights and options. I also cached the index-keyword retriever object so enabling it for some plans does not repeatedly reload its JSON inputs.

A second area of progress is evaluation support. I updated the benchmark path to run through the planner-aware execution path instead of bypassing it. Benchmark results now record planner mode, predicted query category, planner metadata, and per-query latency. I also extended the HTML reporting utility so the generated summaries expose query categories and latency values alongside answer-quality metrics. This is important for the final project because my central claim is not only that adaptive planning changes behavior, but that it improves the quality-latency tradeoff. The checkpoint version does not yet implement a learned cost model, but it does establish the measurement path needed for that later step.

I added unit coverage for the planner itself. The new tests verify that representative questions are classified into the intended categories, that the category-to-configuration mapping changes the expected parts of the pipeline, and that planner metadata is propagated into the resolved query configuration. This matters because the planner is now a first-class system component, not just a convenience function. If I change heuristics later while building the 100% version, I will have regression coverage for the behavior I am using at the checkpoint.

Beyond the planner, I also made supporting runtime fixes that were necessary once I began using the system interactively with planner-selected retrieval settings. I found that follow-up chat queries could overflow the generator model’s context window when long retrieval context and conversational rewriting combined in a single prompt. I addressed that by adding prompt-budget handling in generation so retrieved chunks are trimmed to fit within the actual model context before completion is requested. I also fixed a reranking bug where the cross-encoder path returned `(chunk, score)` tuples instead of just chunk text, and a logging mismatch where the CLI logger was recording the entire chunk corpus rather than the selected retrieved set. These are not the main research contribution, but they were necessary to keep the planned pipeline stable enough to evaluate.

At this point, the working checkpoint system includes the following main components: document indexing over the course textbook, hybrid retrieval with FAISS and BM25, optional index-keyword retrieval, ensemble ranking, optional cross-encoder reranking, local generation with Qwen GGUF models, heuristic query planning, benchmark execution, and planner-aware logging/reporting. In other words, the main 75% architecture from the proposal is now present and integrated end to end. The missing pieces are mostly in the cost-modeling and adaptive-feedback layers planned for the later milestones.

## Challenges and Observations

The biggest challenge so far is that adapting a RAG pipeline is not just a modeling problem; it is also a systems-integration problem. The codebase originally assumed that one global configuration would be used for all queries. Once I moved to per-query planning, several hidden assumptions surfaced. Retrievers and rankers had to be re-instantiated in a lightweight way, planner decisions had to be logged, and benchmark execution had to use the same path as interactive chat. This work took longer than simply writing the classifier, but it was necessary to make the checkpoint meaningful.

Another observation is that local models make latency-quality tradeoffs very visible. On-device generation and reranking make it easy to see why a static pipeline is wasteful. The expensive settings are not always justified for short factual questions, while analytical questions benefit much more from broader retrieval and reranking. This supports the main premise of my project. At the same time, my current planner is still hand-tuned. It reflects reasonable heuristics, but it is not yet backed by a learned cost model or a sufficiently large benchmark set. That means the system can now adapt, but the adaptation is still rule-based rather than data-driven.

I also observed that evaluation remains a limitation. The current benchmark infrastructure is solid enough for comparison, but the question set still needs to be expanded if I want strong claims across all query categories. The proposal calls for a larger benchmark suite, and that is still necessary. Right now, the checkpoint demonstrates architecture, integration, and initial instrumentation more strongly than final empirical results.

## Next Steps

My next step is to move from rule-based planning to cost-based selection. The infrastructure I added in this checkpoint gives me the right starting point: per-query plan resolution, category-aware execution, latency tracking, and benchmark reporting. I plan to define a plan space over retrieval mode, ranker weights, candidate pool size, HyDE, reranking, and double prompting. Then I will compare candidate plans using observed latency and answer-quality measurements from the benchmark suite.

I also need to extend the benchmark workload from the current small set to a larger question set that more evenly covers definitional, explanatory, analytical, and procedural questions. That will make the planner evaluation more credible and will support simple calibration of a cost model. After that, I want to add a budget-aware plan selector that can choose among candidate plans under a latency target, which is the main 100% milestone from the proposal.

If time permits after that, I will work on the feedback-driven portion of the project: lightweight quality estimation, online adjustment of plan preferences, and plan caching for repeated query classes. That is beyond the checkpoint, but the current system has been restructured so those later features can be added cleanly rather than bolted on.

## Appendix: Learning Episode

Topic: Concurrency control through conflict serializability and two-phase locking.

Question 1: What does it mean for two schedules to be conflict equivalent?

Correct textbook chunk(s): If a schedule S can be transformed into a schedule S′ by a series of swaps of non-conflicting instructions, we say that S and S′ are conflict equivalent.

Question 2: When is a schedule conflict serializable?

Correct textbook chunk(s): We say that a schedule S is conflict serializable if it is conflict equivalent to a serial schedule.

Question 3: How does a precedence graph represent ordering constraints between transactions in a schedule?

Correct textbook chunk(s): Consider a schedule S. We construct a directed graph, called a precedence graph, from S. This graph consists of a pair G = (V, E), where V is a set of vertices and E is a set of edges. The set of vertices consists of all the transactions participating in the schedule. The set of edges consists of all edges Ti → Tj for which one of three conditions holds:
1. Ti executes write(Q) before Tj executes read(Q).
2. Ti executes read(Q) before Tj executes write(Q).
3. Ti executes write(Q) before Tj executes write(Q).
If an edge Ti → Tj exists in the precedence graph, then, in any serial schedule S' equivalent to S, Ti must appear before Tj.

Question 4: What are the growing and shrinking phases of the two-phase locking protocol?

Correct textbook chunk(s): One protocol that ensures serializability is the two-phase locking protocol. This protocol requires that each transaction issue lock and unlock requests in two phases:
1. Growing phase. A transaction may obtain locks, but may not release any lock.
2. Shrinking phase. A transaction may release locks, but may not obtain any new locks.

Question 5: How does strict two-phase locking prevent cascading rollbacks?

Correct textbook chunk(s): Cascading rollbacks can be avoided by a modification of two-phase locking called the strict two-phase locking protocol. This protocol requires not only that locking be two phase, but also that all exclusive-mode locks taken by a transaction be held until that transaction commits. This requirement ensures that any data written by an uncommitted transaction are locked in exclusive mode until the transaction commits, preventing any other transaction from reading the data.
