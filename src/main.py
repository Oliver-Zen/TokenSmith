# noinspection PyUnresolvedReferences
import faiss  # force single OpenMP init

import argparse
import json
import pathlib
import sys
from typing import Dict, Optional, List, Tuple, Union, Any

from rich.live import Live
from rich.console import Console
from rich.markdown import Markdown

from src.config import RAGConfig
from src.generator import answer, double_answer, dedupe_generated_text
from src.index_builder import build_index
from src.instrumentation.logging import get_logger
from src.planning.heuristics import HeuristicQueryPlanner
from src.ranking.ranker import EnsembleRanker
from src.preprocessing.chunking import DocumentChunker
from src.query_enhancement import generate_hypothetical_document, contextualize_query
from src.retriever import (
    filter_retrieved_chunks, 
    BM25Retriever, 
    FAISSRetriever, 
    IndexKeywordRetriever, 
    get_page_numbers, 
    load_artifacts
)
from src.ranking.reranker import rerank

ANSWER_NOT_FOUND = "I'm sorry, but I don't have enough information to answer that question."


def build_query_planner(cfg: Any):
    planner_mode = getattr(cfg, "planner_mode", "none")
    if not isinstance(planner_mode, str):
        planner_mode = "none"
    if planner_mode.lower() == "heuristic":
        return HeuristicQueryPlanner(cfg)
    return None


def resolve_query_config(question: str, cfg: Any, planner: Any = None) -> Tuple[Any, Dict[str, Any]]:
    planner_mode = getattr(cfg, "planner_mode", "none")
    if not isinstance(planner_mode, str):
        planner_mode = "none"
    if planner_mode.lower() == "none":
        return cfg, {}

    planner = planner or build_query_planner(cfg)
    if planner is None:
        return cfg, {}

    effective_cfg = planner.plan(question)
    decision = planner.last_decision
    planner_info = {
        "planner_enabled": True,
        "planner_name": planner.name,
        "query_category": decision.category if decision else None,
        "planner_features": decision.features if decision else {},
        "planner_rationale": decision.rationale if decision else [],
        "planner_config_diff": decision.config_diff if decision else {},
    }
    return effective_cfg, planner_info


def build_query_runtime(cfg: Any, artifacts: Dict[str, Any]) -> Tuple[List[Any], EnsembleRanker]:
    if (
        artifacts.get("retrievers") is not None
        and artifacts.get("ranker") is not None
        and artifacts.get("faiss_index") is None
        and artifacts.get("bm25_index") is None
    ):
        return artifacts["retrievers"], artifacts["ranker"]

    retrievers = [
        FAISSRetriever(artifacts["faiss_index"], cfg.embed_model),
        BM25Retriever(artifacts["bm25_index"]),
    ]

    if cfg.ranker_weights.get("index_keywords", 0) > 0:
        cache_key = (str(cfg.extracted_index_path), str(cfg.page_to_chunk_map_path))
        index_retriever_cache = artifacts.setdefault("_index_keyword_retriever_cache", {})
        if cache_key not in index_retriever_cache:
            index_retriever_cache[cache_key] = IndexKeywordRetriever(
                cfg.extracted_index_path,
                cfg.page_to_chunk_map_path,
            )
        retrievers.append(index_retriever_cache[cache_key])

    ranker = EnsembleRanker(
        ensemble_method=cfg.ensemble_method,
        weights=cfg.ranker_weights,
        rrf_k=int(cfg.rrf_k),
    )
    return retrievers, ranker


def retrieve_and_rank_chunks(
    question: str,
    cfg: Any,
    artifacts: Dict[str, Any],
    is_test_mode: bool = False,
) -> Tuple[List[str], List[int], List[float], Optional[List[Dict[str, Any]]], Optional[str]]:
    chunks = artifacts["chunks"]
    retrievers, ranker = build_query_runtime(cfg, artifacts)
    retrieval_query = question
    hyde_query = None

    if cfg.use_hyde:
        hyde_query = generate_hypothetical_document(
            question,
            cfg.gen_model,
            max_tokens=cfg.hyde_max_tokens,
        )
        retrieval_query = hyde_query

    pool_n = max(cfg.num_candidates, cfg.top_k + 10)
    raw_scores: Dict[str, Dict[int, float]] = {}
    for retriever in retrievers:
        raw_scores[retriever.name] = retriever.get_scores(retrieval_query, pool_n, chunks)

    ordered, scores = ranker.rank(raw_scores=raw_scores)
    topk_idxs = filter_retrieved_chunks(cfg, chunks, ordered)
    ranked_chunks = [chunks[i] for i in topk_idxs]

    chunks_info = None
    if is_test_mode:
        faiss_scores = raw_scores.get("faiss", {})
        bm25_scores = raw_scores.get("bm25", {})
        index_scores = raw_scores.get("index_keywords", {})

        faiss_ranked = sorted(faiss_scores.keys(), key=lambda i: faiss_scores[i], reverse=True)
        bm25_ranked = sorted(bm25_scores.keys(), key=lambda i: bm25_scores[i], reverse=True)
        index_ranked = sorted(index_scores.keys(), key=lambda i: index_scores[i], reverse=True)

        faiss_ranks = {idx: rank + 1 for rank, idx in enumerate(faiss_ranked)}
        bm25_ranks = {idx: rank + 1 for rank, idx in enumerate(bm25_ranked)}
        index_ranks = {idx: rank + 1 for rank, idx in enumerate(index_ranked)}

        chunks_info = []
        for rank, idx in enumerate(topk_idxs, 1):
            chunks_info.append({
                "rank": rank,
                "chunk_id": idx,
                "content": chunks[idx],
                "faiss_score": faiss_scores.get(idx, 0),
                "faiss_rank": faiss_ranks.get(idx, 0),
                "bm25_score": bm25_scores.get(idx, 0),
                "bm25_rank": bm25_ranks.get(idx, 0),
                "index_score": index_scores.get(idx, 0),
                "index_rank": index_ranks.get(idx, 0),
            })

    ranked_chunks = rerank(question, ranked_chunks, mode=cfg.rerank_mode, top_n=cfg.rerank_top_k)
    return ranked_chunks, topk_idxs, scores, chunks_info, hyde_query

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Welcome to TokenSmith!")
    parser.add_argument("mode", choices=["index", "chat"], help="operation mode")
    parser.add_argument("--pdf_dir", default="data/chapters/", help="directory containing PDF files")
    parser.add_argument("--index_prefix", default="textbook_index", help="prefix for generated index files")
    parser.add_argument("--model_path", help="path to generation model")
    parser.add_argument("--system_prompt_mode", choices=["baseline", "tutor", "concise", "detailed"], default="baseline")
    
    indexing_group = parser.add_argument_group("indexing options")
    indexing_group.add_argument("--keep_tables", action="store_true")
    indexing_group.add_argument("--multiproc_indexing", action="store_true")
    indexing_group.add_argument("--embed_with_headings", action="store_true")
    parser.add_argument(
        "--double_prompt",
        action="store_true",
        help="enable double prompting for higher quality answers"
    )

    return parser.parse_args()

def run_index_mode(args: argparse.Namespace, cfg: RAGConfig):
    strategy = cfg.get_chunk_strategy()
    chunker = DocumentChunker(strategy=strategy, keep_tables=args.keep_tables)
    artifacts_dir = cfg.get_artifacts_directory()

    data_dir = pathlib.Path("data")
    print(f"Looking for markdown files in {data_dir.resolve()}...")
    md_files = sorted(data_dir.glob("*.md"))
    print(f"Found {len(md_files)} markdown files.")
    print(f"First 5 markdown files: {[str(f) for f in md_files[:5]]}")

    if not md_files:
        print("ERROR: No markdown files found in data/.", file=sys.stderr)
        sys.exit(1)

    build_index(
        markdown_file=str(md_files[0]),
        chunker=chunker,
        chunk_config=cfg.chunk_config,
        embedding_model_path=cfg.embed_model,
        artifacts_dir=artifacts_dir,
        index_prefix=args.index_prefix,
        use_multiprocessing=args.multiproc_indexing,
        use_headings=args.embed_with_headings,
    )

def use_indexed_chunks(question: str, chunks: list) -> list:
    # Logic for keyword matching from textbook index
    try:
        with open('index/sections/textbook_index_page_to_chunk_map.json', 'r') as f:
            page_to_chunk_map = json.load(f)
        with open('data/extracted_index.json', 'r') as f:
            extracted_index = json.load(f)
    except FileNotFoundError:
        return []

    keywords = get_keywords(question)
    chunk_ids = {
        chunk_id
        for word in keywords
        if word in extracted_index
        for page_no in extracted_index[word]
        for chunk_id in page_to_chunk_map.get(str(page_no), [])
    }
    return [chunks[cid] for cid in chunk_ids], list(chunk_ids)

def get_answer(
    question: str,
    cfg: RAGConfig,
    args: argparse.Namespace,
    logger: Any,
    console: Optional["Console"],
    artifacts: Optional[Dict] = None,
    golden_chunks: Optional[list] = None,
    is_test_mode: bool = False,
    additional_log_info: Optional[Dict[str, Any]] = None
) -> Union[str, Tuple[str, List[Dict[str, Any]], Optional[str]]]:
    """
    Run a single query through the pipeline.
    """
    effective_cfg, planner_info = resolve_query_config(question, cfg, artifacts.get("planner") if artifacts else None)
    effective_log_info = dict(additional_log_info or {})
    effective_log_info.update(planner_info)

    chunks = artifacts["chunks"]
    sources = artifacts["sources"]
    # Ensure these locals exist for all control flows to avoid UnboundLocalError
    ranked_chunks: List[str] = []
    topk_idxs: List[int] = []
    scores = []
    
    # Step 1: Get chunks (golden, retrieved, or none)
    chunks_info = None
    hyde_query = None
    if golden_chunks and effective_cfg.use_golden_chunks:
        # Use provided golden chunks
        ranked_chunks = golden_chunks
    elif effective_cfg.disable_chunks:
        # No chunks - baseline mode
        ranked_chunks = []
    elif effective_cfg.use_indexed_chunks:
        ranked_chunks, topk_idxs = use_indexed_chunks(question, chunks)
    else:
        ranked_chunks, topk_idxs, scores, chunks_info, hyde_query = retrieve_and_rank_chunks(
            question=question,
            cfg=effective_cfg,
            artifacts=artifacts,
            is_test_mode=is_test_mode,
        )

    if not ranked_chunks and not effective_cfg.disable_chunks:
        if console:
            console.print(f"\n{ANSWER_NOT_FOUND}\n")
        return ANSWER_NOT_FOUND

    # Step 4: Generation
    model_path = effective_cfg.gen_model
    system_prompt = args.system_prompt_mode or effective_cfg.system_prompt_mode

    use_double = getattr(args, "double_prompt", False) or effective_cfg.use_double_prompt

    if use_double:
        stream_iter = double_answer(
            question,
            ranked_chunks,
            model_path,
            max_tokens=effective_cfg.max_gen_tokens,
            system_prompt_mode=system_prompt,
        )
    else:
        stream_iter = answer(
            question,
            ranked_chunks,
            model_path,
            max_tokens=effective_cfg.max_gen_tokens,
            system_prompt_mode=system_prompt,
        )

    if is_test_mode:
        # We do not render MD in the test mode
        ans = ""
        for delta in stream_iter:
            ans += delta
        ans = dedupe_generated_text(ans)
        return ans, chunks_info, hyde_query
    else:
        # Accumulate the full text while rendering incremental Markdown chunks
        ans = render_streaming_ans(console, stream_iter)

        # Logging
        meta = artifacts.get("meta", [])
        log_count = min(
            len(topk_idxs),
            len(ranked_chunks) if ranked_chunks else len(topk_idxs),
            len(scores) if scores else len(topk_idxs),
        )
        log_top_idxs = topk_idxs[:log_count]
        log_chunks = [chunks[i] for i in log_top_idxs]
        log_sources = [sources[i] for i in log_top_idxs]
        page_nums = get_page_numbers(log_top_idxs, meta)
        logger.save_chat_log(
            query=question,
            config_state=effective_cfg.get_config_state(),
            ordered_scores=scores[:log_count] if scores else [],
            chat_request_params={
                "system_prompt": system_prompt,
                "max_tokens": effective_cfg.max_gen_tokens
            },
            top_idxs=log_top_idxs,
            chunks=log_chunks,
            sources=log_sources,
            page_map=page_nums,
            full_response=ans,
            top_k=len(log_top_idxs),
            additional_log_info=effective_log_info
        )
        return ans

def render_streaming_ans(console, stream_iter):
    ans = ""
    is_first = True
    with Live(console=console, refresh_per_second=8) as live:
        for delta in stream_iter:
            if is_first:
                console.print("\n[bold cyan]=== START OF ANSWER ===[/bold cyan]\n")
                is_first = False
            ans += delta
            live.update(Markdown(ans))
    ans = dedupe_generated_text(ans)
    live.update(Markdown(ans))
    console.print("\n[bold cyan]=== END OF ANSWER ===[/bold cyan]\n")
    return ans

def get_keywords(question: str) -> list:
    """
    Simple keyword extraction from the question.
    """
    stopwords = set([
        "the", "is", "at", "which", "on", "for", "a", "an", "and", "or", "in", 
        "to", "of", "by", "with", "that", "this", "it", "as", "are", "was", "what"
    ])
    words = question.lower().split()
    keywords = [word.strip('.,!?()[]') for word in words if word not in stopwords]
    return keywords

def run_chat_session(args: argparse.Namespace, cfg: RAGConfig):
    logger = get_logger()
    console = Console()
    planner = build_query_planner(cfg)

    print("Initializing TokenSmith Chat...")
    try:
        artifacts_dir = cfg.get_artifacts_directory()
        faiss_idx, bm25_idx, chunks, sources, meta = load_artifacts(artifacts_dir, args.index_prefix)
        print(f"Loaded {len(chunks)} chunks and {len(sources)} sources from artifacts.")
        print("Loaded indexes and source artifacts.")
        artifacts = {
            "chunks": chunks,
            "sources": sources,
            "faiss_index": faiss_idx,
            "bm25_index": bm25_idx,
            "meta": meta,
            "planner": planner,
        }
    except Exception as e:
        print(f"ERROR: {e}. Run 'index' mode first.")
        sys.exit(1)

    chat_history = []
    print("Initialization complete. You can start asking questions!")
    print("Type 'exit' or 'quit' to end the session.")
    while True:
        print("CHAT HISTORY:", chat_history)  # Debug print to trace chat history
        try:
            q = input("\nAsk > ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            
            effective_q = q
            additional_log_info = {}
            if cfg.enable_history and chat_history:
                try:
                    effective_q = contextualize_query(q, chat_history, cfg.gen_model)
                    additional_log_info["is_contextualizing_query"] = True
                    additional_log_info["contextualized_query"] = effective_q
                    additional_log_info["original_query"] = q
                    additional_log_info["chat_history"] = chat_history
                    print(f"Contextualized Query: {effective_q}")  # Debug print to trace contextualization
                except Exception as e:
                    print(f"Warning: Failed to contextualize query: {e}. Using original query.")
                    effective_q = q
            
            # Use the single query function. get_answer also renders the streaming markdown and takes care of logging, so we need not do anything else here.
            ans = get_answer(effective_q, cfg, args, logger, console, artifacts=artifacts, additional_log_info=additional_log_info)

            # Update Chat history (make it atomic for user + assistant turn)
            try:
                user_turn      = {"role": "user", "content": q}
                assistant_turn = {"role": "assistant", "content": ans}
                chat_history  += [user_turn, assistant_turn]
            except Exception as e:
                print(f"Warning: Failed to update chat history: {e}")
                # We can continue without chat history, so we do not break the loop here.

            # Trim chat history to avoid exceeding context window
            if len(chat_history) > cfg.max_history_turns * 2:
                chat_history = chat_history[-cfg.max_history_turns * 2:]

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            break



def main():
    args = parse_args()
    config_path = pathlib.Path("config/config.yaml")
    if not config_path.exists(): raise FileNotFoundError("config/config.yaml not found.")
    cfg = RAGConfig.from_yaml(config_path)
    print(f"Loaded configuration from {config_path.resolve()}.")
    if args.mode == "index":
        run_index_mode(args, cfg)
    elif args.mode == "chat":
        run_chat_session(args, cfg)

if __name__ == "__main__":
    main()
