#!/usr/bin/env python3
"""
Latency Summarizer Script

Analyzes JSONL logs to compute latency statistics (p50, p95, mean, min, max)
for each pipeline stage: retrieval, ranking, generation, and end-to-end.

Usage:
    python scripts/summarize_latency.py                    # Latest session
    python scripts/summarize_latency.py SESSION_ID         # Specific session
    python scripts/summarize_latency.py --all              # All sessions
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from statistics import mean, median, stdev


def load_session_logs(log_file: Path) -> List[Dict[str, Any]]:
    """Load all log entries from a JSONL file."""
    if not log_file.exists():
        return []

    logs = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return logs


def extract_latency_data(logs: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Extract latency timings from query logs."""
    latencies = {
        "retrieval": [],
        "ranking": [],
        "generation": [],
        "total": []
    }

    for log in logs:
        if log.get("event") == "query" and "latency" in log:
            timing = log["latency"]
            if "retrieval_seconds" in timing:
                latencies["retrieval"].append(timing["retrieval_seconds"])
            if "ranking_seconds" in timing:
                latencies["ranking"].append(timing["ranking_seconds"])
            if "generation_seconds" in timing:
                latencies["generation"].append(timing["generation_seconds"])
            if "total_seconds" in timing:
                latencies["total"].append(timing["total_seconds"])

    return latencies


def compute_percentile(data: List[float], percentile: int) -> float:
    """Compute the Nth percentile of a list of numbers."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = (percentile / 100) * (len(sorted_data) - 1)
    lower = int(index)
    upper = lower + 1
    weight = index - lower

    if upper >= len(sorted_data):
        return sorted_data[lower]
    return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight


def compute_statistics(data: List[float]) -> Dict[str, float]:
    """Compute comprehensive statistics for a dataset."""
    if not data:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0
        }

    return {
        "count": len(data),
        "mean": mean(data),
        "median": median(data),
        "p50": compute_percentile(data, 50),
        "p95": compute_percentile(data, 95),
        "p99": compute_percentile(data, 99),
        "min": min(data),
        "max": max(data),
        "std": stdev(data) if len(data) > 1 else 0.0
    }


def format_duration(seconds: float) -> str:
    """Format duration in seconds to a readable string."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f}μs"
    elif seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    else:
        return f"{seconds:.2f}s"


def print_statistics_table(stats: Dict[str, Dict[str, float]]):
    """Print latency statistics in a formatted table."""
    print("\n" + "="*80)
    print("  LATENCY STATISTICS (Per-Stage Breakdown)")
    print("="*80)
    print(f"{'Stage':<15} {'Count':>8} {'Mean':>10} {'Median':>10} {'P50':>10} {'P95':>10} {'P99':>10} {'Min':>10} {'Max':>10}")
    print("-"*80)

    stages = ["retrieval", "ranking", "generation", "total"]
    stage_names = {
        "retrieval": "Retrieval",
        "ranking": "Ranking",
        "generation": "Generation",
        "total": "End-to-End"
    }

    for stage in stages:
        if stage in stats and stats[stage]["count"] > 0:
            s = stats[stage]
            print(f"{stage_names[stage]:<15} {s['count']:>8} "
                  f"{format_duration(s['mean']):>10} "
                  f"{format_duration(s['median']):>10} "
                  f"{format_duration(s['p50']):>10} "
                  f"{format_duration(s['p95']):>10} "
                  f"{format_duration(s['p99']):>10} "
                  f"{format_duration(s['min']):>10} "
                  f"{format_duration(s['max']):>10}")

    print("="*80 + "\n")


def print_summary(stats: Dict[str, Dict[str, float]], session_info: Dict[str, Any]):
    """Print a summary of key metrics."""
    print(f"\n📊 Session Summary")
    print(f"   Session ID: {session_info.get('session_id', 'unknown')}")
    print(f"   Queries Processed: {stats['total']['count']}")

    if stats['total']['count'] > 0:
        print(f"\n⏱️  Key Latency Metrics:")
        print(f"   End-to-End P50: {format_duration(stats['total']['p50'])}")
        print(f"   End-to-End P95: {format_duration(stats['total']['p95'])}")
        print(f"   Mean Total Time: {format_duration(stats['total']['mean'])}")

        # Breakdown by stage
        if all(stage in stats and stats[stage]['count'] > 0 for stage in ['retrieval', 'ranking', 'generation']):
            print(f"\n📈 Stage Breakdown (Mean):")
            total_mean = stats['total']['mean']
            print(f"   Retrieval:   {format_duration(stats['retrieval']['mean'])} ({stats['retrieval']['mean']/total_mean*100:.1f}%)")
            print(f"   Ranking:     {format_duration(stats['ranking']['mean'])} ({stats['ranking']['mean']/total_mean*100:.1f}%)")
            print(f"   Generation:  {format_duration(stats['generation']['mean'])} ({stats['generation']['mean']/total_mean*100:.1f}%)")


def analyze_session(log_file: Path) -> bool:
    """Analyze latency for a single session."""
    print(f"\n🔍 Analyzing: {log_file.name}")

    logs = load_session_logs(log_file)
    if not logs:
        print(f"   ❌ No logs found in {log_file}")
        return False

    # Extract session info
    session_info = next((log for log in logs if log.get("event") == "session_start"), {})

    # Extract latency data
    latencies = extract_latency_data(logs)

    # Check if any latency data exists
    if not any(latencies.values()):
        print(f"   ⚠️  No latency data found. Enable latency logging with --latency-logging flag or set enable_latency_logging: true in config.")
        return False

    # Compute statistics
    stats = {stage: compute_statistics(data) for stage, data in latencies.items()}

    # Print results
    print_statistics_table(stats)
    print_summary(stats, session_info)

    return True


def find_latest_session() -> Path:
    """Find the most recent session log file."""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return None

    log_files = list(logs_dir.glob("run_*.jsonl"))
    if not log_files:
        return None

    return max(log_files, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize latency statistics from TokenSmith JSONL logs"
    )
    parser.add_argument(
        "session_id",
        nargs="?",
        help="Session ID to analyze (default: latest session)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all sessions"
    )

    args = parser.parse_args()

    logs_dir = Path("logs")
    if not logs_dir.exists():
        print("❌ No logs directory found. Run TokenSmith in chat mode first.")
        sys.exit(1)

    if args.all:
        # Analyze all sessions
        log_files = sorted(logs_dir.glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not log_files:
            print("❌ No log files found in logs/")
            sys.exit(1)

        print(f"\n📚 Found {len(log_files)} session(s)")
        success_count = 0
        for log_file in log_files:
            if analyze_session(log_file):
                success_count += 1

        print(f"\n✅ Successfully analyzed {success_count}/{len(log_files)} session(s)")

    elif args.session_id:
        # Analyze specific session
        log_file = logs_dir / f"run_{args.session_id}.jsonl"
        if not log_file.exists():
            print(f"❌ Log file not found: {log_file}")
            sys.exit(1)

        if not analyze_session(log_file):
            sys.exit(1)

    else:
        # Analyze latest session
        log_file = find_latest_session()
        if not log_file:
            print("❌ No log files found in logs/")
            sys.exit(1)

        if not analyze_session(log_file):
            sys.exit(1)


if __name__ == "__main__":
    main()
