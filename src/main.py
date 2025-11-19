import argparse
import pathlib
import sys
import time
from typing import Dict, Optional

from src.config import QueryPlanConfig
from src.generator import answer
from src.index_builder import build_index
from src.instrumentation.logging import init_logger, get_logger, RunLogger
from src.ranking.ranker import EnsembleRanker
from src.preprocessing.chunking import DocumentChunker
from src.retriever import apply_seg_filter, BM25Retriever, FAISSRetriever, load_artifacts, format_citations


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the application."""
    parser = argparse.ArgumentParser(
        description="Welcome to TokenSmith!"
    )

    # Required arguments
    parser.add_argument(
        "mode",
        choices=["index", "chat"],
        help="operation mode: 'index' to build index, 'chat' to query"
    )

    # Common arguments
    parser.add_argument(
        "--pdf_dir",
        default="data/chapters/",
        help="directory containing PDF files (default: %(default)s)"
    )
    parser.add_argument(
        "--index_prefix",
        default="textbook_index",
        help="prefix for generated index files (default: %(default)s)"
    )
    parser.add_argument(
        "--model_path",
        help="path to generation model (uses config default if not specified)"
    )
    parser.add_argument(
        "--system_prompt_mode",
        choices=["baseline", "tutor", "concise", "detailed"],
        default="baseline",
        help="system prompt mode (choices: baseline, tutor, concise, detailed)"
    )
    parser.add_argument(
        "--citations",
        action="store_true",
        help="enable source citations in answers (default: uses config value)"
    )
    parser.add_argument(
        "--latency-logging",
        action="store_true",
        help="enable detailed latency logging for retrieval, ranking, and generation stages (default: uses config value)"
    )

    # Indexing-specific arguments
    indexing_group = parser.add_argument_group("indexing options")
    indexing_group.add_argument(
        "--pdf_range",
        metavar="START-END",
        help="specific range of PDFs to index (e.g., '27-33')"
    )
    indexing_group.add_argument(
        "--keep_tables",
        action="store_true",
        help="include tables in the index"
    )
    indexing_group.add_argument(
        "--visualize",
        action="store_true",
        help="generate visualizations during indexing"
    )

    return parser.parse_args()


def run_index_mode(args: argparse.Namespace, cfg: QueryPlanConfig):
    """Handles the logic for building the index."""

    # Robust range filtering
    try:
        if args.pdf_range:
            start, end = map(int, args.pdf_range.split("-"))
            pdf_paths = [f"{i}.pdf" for i in range(start, end + 1)] # Inclusive range
            print(f"Indexing PDFs in range: {start}-{end}")
        else:
            pdf_paths = None
    except ValueError:
        print(f"ERROR: Invalid format for --pdf_range. Expected 'start-end', but got '{args.pdf_range}'.")
        sys.exit(1)
    
    strategy = cfg.make_strategy()
    chunker = DocumentChunker(strategy=strategy, keep_tables=args.keep_tables)
    
    artifacts_dir = cfg.make_artifacts_directory()

    build_index(
        markdown_file="data/book_without_image.md",
        chunker=chunker,
        chunk_config=cfg.chunk_config,
        embedding_model_path=cfg.embed_model,
        artifacts_dir=artifacts_dir,
        index_prefix=args.index_prefix,
        do_visualize=args.visualize,
    )


def get_answer(
    question: str,
    cfg: QueryPlanConfig,
    args: argparse.Namespace,
    logger: "RunLogger",
    artifacts: Optional[Dict] = None,
    golden_chunks: Optional[list] = None
) -> str:
    """
    Run a single query through the pipeline.
    """
    chunks = artifacts["chunks"]
    sources = artifacts["sources"]
    metadata = artifacts["metadata"]
    retrievers = artifacts["retrievers"]
    ranker = artifacts["ranker"]

    logger.log_query_start(question)

    # Determine if citations and latency logging should be enabled (CLI overrides config)
    enable_citations = getattr(args, 'citations', False) or getattr(cfg, 'enable_citations', False)
    enable_latency = getattr(args, 'latency_logging', False) or getattr(cfg, 'enable_latency_logging', False)

    # Initialize timing data (only if latency logging is enabled)
    timings = {}
    if enable_latency:
        overall_start = time.perf_counter()

    # Step 1: Get chunks (golden, retrieved, or none)
    topk_idxs = []
    if golden_chunks and cfg.use_golden_chunks:
        # Use provided golden chunks
        ranked_chunks = golden_chunks
    elif cfg.disable_chunks:
        # No chunks - baseline mode
        ranked_chunks = []
    else:
        # Step 1: Retrieval
        if enable_latency:
            retrieval_start = time.perf_counter()

        pool_n = max(cfg.pool_size, cfg.top_k + 10)
        raw_scores: Dict[str, Dict[int, float]] = {}
        for retriever in retrievers:
            raw_scores[retriever.name] = retriever.get_scores(question, pool_n, chunks)

        if enable_latency:
            timings["retrieval_seconds"] = time.perf_counter() - retrieval_start
        # TODO: Fix retrieval logging.

        # Step 2: Ranking
        if enable_latency:
            ranking_start = time.perf_counter()

        ordered = ranker.rank(raw_scores=raw_scores)
        topk_idxs = apply_seg_filter(cfg, chunks, ordered)
        logger.log_chunks_used(topk_idxs, chunks, sources)

        if enable_latency:
            timings["ranking_seconds"] = time.perf_counter() - ranking_start

        ranked_chunks = [chunks[i] for i in topk_idxs]

        # Step 3: Final Re-ranking (if enabled)
        # Disabled till we fix the core pipeline
        # ranked_chunks = rerank(question, ranked_chunks, mode=cfg.rerank_mode, top_n=cfg.top_k)

    # Step 4: Generation
    if enable_latency:
        generation_start = time.perf_counter()

    model_path = args.model_path or cfg.model_path
    system_prompt = args.system_prompt_mode or cfg.system_prompt_mode

    # Generate answer with optional citations
    if enable_citations and topk_idxs:
        citations = format_citations(topk_idxs, metadata)
    else:
        citations = ""

    ans = answer(
        question,
        ranked_chunks,
        model_path,
        max_tokens=cfg.max_gen_tokens,
        system_prompt_mode=system_prompt
    )

    if enable_latency:
        timings["generation_seconds"] = time.perf_counter() - generation_start

    # Append citations if available
    if citations:
        ans = f"{ans}\n\n{citations}"

    # Calculate total time and log if enabled
    if enable_latency:
        try:
            total_time = time.perf_counter() - overall_start
            timings["total_seconds"] = total_time
            logger.log_latency(timings)
        except Exception as e:
            # Don't let latency logging errors affect the answer
            print(f"Warning: Failed to log latency: {e}")

    return ans


def run_chat_session(args: argparse.Namespace, cfg: QueryPlanConfig):
    """
    Initializes artifacts and runs the main interactive chat loop.
    """
    logger = get_logger()
    # planner = HeuristicQueryPlanner(cfg)

    # Load artifacts, initialize retrievers and rankers once before the loop.
    print("Welcome to Tokensmith! Initializing chat...")
    try:
        # Disabled till we fix the core pipeline
        # cfg = planner.plan(q)
        artifacts_dir = cfg.make_artifacts_directory()
        faiss_index, bm25_index, chunks, sources, metadata = load_artifacts(
            artifacts_dir=artifacts_dir,
            index_prefix=args.index_prefix
        )

        retrievers = [
            FAISSRetriever(faiss_index, cfg.embed_model),
            BM25Retriever(bm25_index)
        ]
        ranker = EnsembleRanker(
            ensemble_method=cfg.ensemble_method,
            weights=cfg.ranker_weights,
            rrf_k=int(cfg.rrf_k)
        )

        # Package artifacts for reuse
        artifacts = {
            "chunks": chunks,
            "sources": sources,
            "metadata": metadata,
            "retrievers": retrievers,
            "ranker": ranker
        }
    except Exception as e:
        print(f"ERROR: Failed to initialize chat artifacts: {e}")
        print("Please ensure you have run 'index' mode first.")
        sys.exit(1)

    print("Initialization complete. You can start asking questions!")
    print("Type 'exit' or 'quit' to end the session.")
    while True:
        try:
            q = input("\nAsk > ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            # Use the single query function
            ans = get_answer(q, cfg, args, logger=logger,artifacts=artifacts)

            print("\n=================== START OF ANSWER ===================")
            print(ans.strip() if ans and ans.strip() else "(No output from model)")
            print("\n==================== END OF ANSWER ====================")
            logger.log_generation(ans, {"max_tokens": cfg.max_gen_tokens, "model_path": args.model_path or cfg.model_path})
            logger.log_query_complete()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            logger.log_error(str(e))
            break

    # TODO: Fix completion logging.
    # logger.log_query_complete()


def main():
    """Main entry point for the script."""
    args = parse_args()

    # Config loading
    config_path = pathlib.Path("config/config.yaml")
    cfg = None
    if config_path.exists():
        cfg = QueryPlanConfig.from_yaml(config_path)

    if cfg is None:
        raise FileNotFoundError(
            "No config file provided and no fallback found at config/ or ~/.config/tokensmith/"
        )

    init_logger(cfg)

    if args.mode == "index":
        run_index_mode(args, cfg)
    elif args.mode == "chat":
        run_chat_session(args, cfg)


if __name__ == "__main__":
    main()
