"""
Report Schema
=============
WHAT: Defines the Reviewer Agent's structured verdict on report quality.
HOW:  ReviewResult has status ("approved"/"rewrite"), feedback, missing sections, score.
      The Reviewer uses `with_structured_output(ReviewResult)` to produce this.
WHY:  The conditional edge after the Reviewer checks `status` to decide:
      - "approved" → END (report is final)
      - "rewrite"  → Writer (try again with feedback)
      This structured output drives the graph's control flow.

INTERVIEW Q&A:
  Q: How does the Reviewer control the graph flow?
  A: The Reviewer produces a structured ReviewResult with status "approved" or
     "rewrite." The conditional edge function reads this status and routes to
     either END or back to the Writer. The feedback field contains specific
     instructions for improvement, so the Writer knows exactly what to fix.
     I cap revisions at MAX_REVISIONS (default 3) to prevent infinite loops.
"""

from typing import Literal
from pydantic import BaseModel, Field


class ReviewResult(BaseModel):
    """Structured review verdict from the Reviewer Agent."""

    status: Literal["approved", "rewrite"] = Field(
        description="Whether the report is approved or needs rewriting"
    )
    feedback: str = Field(
        description="Detailed feedback — what needs improvement if rewrite"
    )
    missing_sections: list[str] = Field(
        default_factory=list,
        description="List of any missing or incomplete report sections"
    )
    score: float = Field(
        ge=0.0, le=10.0,
        description="Quality score from 0 (terrible) to 10 (perfect)"
    )


class ReportMetadata(BaseModel):
    """Metadata about a generated report for API responses."""

    job_id: str
    query: str
    status: str
    confidence_score: float = 0.0
    sources_count: int = 0
    revision_count: int = 0
    created_at: str | None = None
    completed_at: str | None = None
