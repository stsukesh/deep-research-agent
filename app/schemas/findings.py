from pydantic import BaseModel, Field


class Finding(BaseModel):
    """A single verified finding extracted from search results."""

    fact: str = Field(description="Clear, concise factual statement")
    source: str = Field(description="Source attribution (e.g., Reuters, Wikipedia)")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score from 0 (unverified) to 1 (highly reliable)"
    )
    topic: str = Field(description="Which research topic this finding belongs to")


class FindingsList(BaseModel):
    """Collection of structured findings from the Extractor Agent."""

    findings: list[Finding] = Field(
        description="List of extracted and validated findings"
    )