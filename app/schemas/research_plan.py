from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    """Structured research plan produced by the Planner Agent."""

    topics: list[str] = Field(
        description="List of 3-7 specific research sub-topics to investigate"
    )
    objectives: list[str] = Field(
        description="Key questions that the research should answer"
    )
    scope: str = Field(
        description="Brief description of the overall research scope and boundaries"
    )