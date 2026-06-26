"""
Evaluation Metrics
==================
WHAT: Tracks performance metrics for every research run.
HOW:  ResearchMetrics dataclass captures timing, counts, and scores.
      collect_metrics() extracts these from the final graph state.
WHY:  Most candidates never evaluate their AI systems. Adding metrics shows
      you think about production concerns: latency, reliability, quality.

METRICS TRACKED:
  - research_time_seconds: Total pipeline execution time
  - tool_calls_count: How many tool invocations (indicates API cost)
  - citations_count: Number of unique sources cited (quality indicator)
  - confidence_average: Mean confidence across all findings
  - revision_count: How many Writer ← Reviewer cycles
  - topics_researched: Number of topics covered
  - findings_count: Total structured findings extracted

INTERVIEW Q&A:
  Q: How do you evaluate the quality of your AI system?
  A: I track quantitative metrics per research job: execution time, tool calls,
     unique citations, average confidence, revision cycles, and findings count.
     These are stored in PostgreSQL and exposed via /metrics endpoint. Over time,
     I can identify patterns — queries with low confidence might need better
     tools, high revision counts suggest prompt improvements are needed.
     
  Q: What would you add for a more advanced evaluation?
  A: Factual accuracy verification (check citations still resolve), user
     satisfaction scoring (thumbs up/down on reports), A/B testing different
     prompts, LLM-as-judge evaluation (use a separate LLM to score report
     quality), and token cost tracking per query for budget optimization.
"""

import time
from dataclasses import dataclass, asdict
from app.graph.state import GraphState


@dataclass
class ResearchMetrics:
    """Performance metrics for a single research run."""

    research_time_seconds: float = 0.0
    tool_calls_count: int = 0
    citations_count: int = 0
    confidence_average: float = 0.0
    revision_count: int = 0
    topics_researched: int = 0
    findings_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON storage."""
        return asdict(self)


def collect_metrics(state: GraphState, start_time: float) -> ResearchMetrics:
    """
    Extract metrics from the final graph state.
    
    Args:
        state: The final graph state after execution
        start_time: time.time() when execution started
    
    Returns:
        ResearchMetrics with all fields populated
    """
    findings = state.get("findings", [])
    search_results = state.get("search_results", [])
    plan = state.get("research_plan", {})

    # Count unique sources from findings
    unique_sources = set(f.get("source", "") for f in findings if f.get("source"))

    return ResearchMetrics(
        research_time_seconds=round(time.time() - start_time, 2),
        tool_calls_count=len(search_results),
        citations_count=len(unique_sources),
        confidence_average=state.get("confidence_score", 0.0),
        revision_count=state.get("revision_count", 0),
        topics_researched=len(plan.get("topics", [])),
        findings_count=len(findings),
    )
