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