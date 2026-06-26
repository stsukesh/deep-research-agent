"""
Planner Agent (Agent 1)
=======================
WHAT: Converts a vague user query into a structured ResearchPlan.
HOW:  Uses ChatGroq with `with_structured_output(ResearchPlan)` to force the LLM
      to return valid JSON matching our Pydantic schema.
WHY:  Vague queries like "Analyze Nvidia" produce bad research. The Planner
      decomposes it into specific, actionable topics like "Revenue Analysis,"
      "AI Product Portfolio," "Competitive Landscape," etc.

FLOW:
  Input:  state["query"] = "Analyze Nvidia's AI business strategy for 2026"
  Output: state["research_plan"] = {"topics": [...], "objectives": [...], "scope": "..."}

INTERVIEW Q&A:
  Q: How do you handle ambiguous user queries?
  A: The Planner Agent uses a system prompt that instructs the LLM to decompose
     any vague query into 3-7 concrete research sub-topics. The LLM is forced to
     output a ResearchPlan Pydantic model via with_structured_output(). If the
     query is too broad, the Planner narrows the scope. If too narrow, it
     expands to cover related dimensions.
     
  Q: What if the LLM returns invalid JSON?
  A: with_structured_output() handles this automatically. For models supporting
     tool calling (like Llama 3.3), it uses the native function-calling API which
     is more reliable. If JSON is still invalid, LangChain retries with an error
     message asking the LLM to fix the format.
"""

from app.agents.llm_factory import get_llm
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import get_settings
from app.schemas.research_plan import ResearchPlan
from app.graph.state import GraphState


PLANNER_SYSTEM_PROMPT = """You are an expert research planner. Your job is to take a user's research query and break it down into a focused research plan.

Instructions:
1. Identify EXACTLY 3 to 5 specific research sub-topics (no more) that are most critical.
2. Define clear objectives — what questions should the research answer?
3. Set the scope — what's included and what's out of bounds?

Think like a senior analyst preparing a tight research brief — quality over quantity.

Examples of good topic breakdowns:
- "Analyze Nvidia" → Revenue & Growth, AI Product Portfolio, Competitive Landscape
- "AI industry trends" → Market Size & Players, Technology Breakthroughs, Regulation & Investment

Always be specific and actionable. Don't use vague topics like "Overview" or "General Info".
Limit to 3-5 topics — more topics slow down research significantly."""


async def planner_node(state: GraphState) -> dict:
    """
    Planner Agent node function.
    
    Reads: state["query"]
    Writes: state["research_plan"], state["messages"], state["current_step"]
    """
    settings = get_settings()
    query = state["query"]

    # Initialize LLM with structured output and fallback support
    llm = get_llm(temperature=0.3)

    # Force LLM to return valid ResearchPlan JSON
    structured_llm = llm.with_structured_output(ResearchPlan)

    # Build messages
    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"Create a detailed research plan for: {query}"),
    ]

    # Invoke and get structured output
    plan: ResearchPlan = await structured_llm.ainvoke(messages)

    return {
        "research_plan": plan.model_dump(),
        "current_step": "planning_complete",
        "messages": [
            HumanMessage(content=f"Research query: {query}"),
            SystemMessage(content=f"Research plan created with {len(plan.topics)} topics: {', '.join(plan.topics)}"),
        ],
    }
