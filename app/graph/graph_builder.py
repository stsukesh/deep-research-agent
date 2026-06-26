from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.graph.state import GraphState
from app.agents.planner import planner_node
from app.agents.researcher import researcher_node
from app.agents.extractor import extractor_node
from app.agents.approval import human_approval_node
from app.agents.writer import writer_node
from app.agents.reviewer import reviewer_node
from app.config import get_settings


def route_after_approval(state: GraphState) -> str:
    if state.get("human_approved", False):
        return "writer"
    return "researcher"


def route_after_review(state: GraphState) -> str:
    settings = get_settings()
    current_step = state.get("current_step", "")
    revision_count = state.get("revision_count", 0)
    if "approved" in current_step:
        return END
    if revision_count >= settings.MAX_REVISIONS:
        return END
    return "writer"


def _build_workflow() -> StateGraph:
    """Build the workflow graph (without compiling — checkpointer added separately)."""
    workflow = StateGraph(GraphState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("approval", human_approval_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "extractor")
    workflow.add_edge("extractor", "approval")

    workflow.add_conditional_edges(
        "approval",
        route_after_approval,
        {"writer": "writer", "researcher": "researcher"},
    )

    workflow.add_edge("writer", "reviewer")

    workflow.add_conditional_edges(
        "reviewer",
        route_after_review,
        {END: END, "writer": "writer"},
    )

    return workflow


async def build_graph_with_postgres():
    """
    Build and compile the graph with AsyncPostgresSaver (Neon).

    Sets up the checkpointer tables on first run, then returns a
    compiled graph ready for use. Call once at app startup.
    """
    settings = get_settings()
    checkpointer = AsyncPostgresSaver.from_conn_string(settings.CHECKPOINT_DB_URL)

    # Create LangGraph checkpoint tables if they don't exist
    await checkpointer.setup()

    workflow = _build_workflow()
    graph = workflow.compile(checkpointer=checkpointer)
    return graph


def build_graph(checkpointer=None):
    """
    Synchronous fallback using MemorySaver (for tests / local dev without DB).
    Production code should use build_graph_with_postgres() instead.
    """
    from langgraph.checkpoint.memory import MemorySaver
    cp = checkpointer or MemorySaver()
    return _build_workflow().compile(checkpointer=cp)