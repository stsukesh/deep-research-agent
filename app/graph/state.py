
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
import operator


class GraphState(TypedDict):
    """
    Shared state flowing through the entire research pipeline.
    Every node reads from and writes to this state.
    """

    # ===== INPUT =====
    query: str  # The user's research query (overwrite)

    # ===== PLANNER OUTPUT =====
    research_plan: dict  # Structured plan from Planner Agent (overwrite)

    # ===== RESEARCHER OUTPUT =====
    # Accumulates search results from ALL tool calls across ALL topics
    search_results: Annotated[list, operator.add]

    # ===== EXTRACTOR OUTPUT =====
    # Accumulates structured findings from extraction
    findings: Annotated[list, operator.add]

    # ===== HUMAN APPROVAL =====
    human_approved: bool  # Whether human approved findings (overwrite)
    human_feedback: str  # Feedback from human if rejected (overwrite)

    # ===== WRITER OUTPUT =====
    report: str  # The generated markdown report (overwrite)

    # ===== REVIEWER OUTPUT =====
    reviewed_report: str  # Final approved report (overwrite)
    review_feedback: str  # Feedback from reviewer if rewrite needed (overwrite)
    review_score: float  # Quality score 0-10 (overwrite)

    # ===== METADATA =====
    confidence_score: float  # Average confidence across findings (overwrite)
    revision_count: int  # Number of revision cycles (overwrite)
    current_step: str  # For UI status tracking (overwrite)

    # ===== MESSAGE HISTORY =====
    # add_messages reducer: accumulates chat messages, handles deduplication
    messages: Annotated[list, add_messages]

    # ===== EVALUATION METRICS =====
    metrics: dict  # Performance metrics (overwrite)
