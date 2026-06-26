"""
ResearchPlan Schema
===================
WHAT: Defines the structured output the Planner Agent MUST produce.
HOW:  `llm.with_structured_output(ResearchPlan)` forces the LLM to return JSON
      matching this Pydantic model. If the LLM returns invalid JSON, LangChain
      retries automatically.
WHY:  In multi-agent systems, downstream agents need PREDICTABLE data structures.
      You can't have the Researcher crash because the Planner returned free-text
      instead of a list of topics.

INTERVIEW Q&A:
  Q: What is structured output and why use Pydantic?
  A: Structured output constrains LLM responses to conform to a schema.
     Pydantic provides runtime validation — if the LLM says confidence is "high"
     instead of 0.92, Pydantic raises a ValidationError. This catches errors at
     the agent boundary, not three steps later when something silently breaks.
     
  Q: How does `with_structured_output` work under the hood?
  A: It injects the JSON schema into the system prompt and instructs the LLM to
     return only valid JSON. For models supporting tool/function calling (like
     Llama 3.3 on Groq), it uses the native function-calling API which is more
     reliable than prompt-based JSON extraction.
"""

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
