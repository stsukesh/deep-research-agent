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