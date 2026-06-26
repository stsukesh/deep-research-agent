"""
Findings Schema
===============
WHAT: Defines structured findings extracted from raw search results.
HOW:  Each Finding has: fact, source, confidence (0-1), and topic attribution.
      The Extractor Agent uses `with_structured_output(FindingsList)` to produce these.
WHY:  Raw search results are messy (HTML, ads, irrelevant text). The Extractor
      transforms them into clean, validated data points. The confidence score
      enables quality filtering — low-confidence findings can be flagged for review.

INTERVIEW Q&A:
  Q: How do you handle varying quality of search results?
  A: Each extracted finding gets a confidence score (0-1) based on source reliability
     and whether the fact is corroborated by multiple sources. I filter out findings
     below 0.3 confidence. The average confidence across all findings also determines
     whether the system flags the research for extra human review.
     
  Q: What's the difference between raw search results and structured findings?
  A: Raw results are what the API returns — URLs, snippets, HTML content. Structured
     findings are extracted facts with source attribution, validated by Pydantic.
     It's the difference between a pile of newspaper clippings and a curated fact sheet.
"""

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
