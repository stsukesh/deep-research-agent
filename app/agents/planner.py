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