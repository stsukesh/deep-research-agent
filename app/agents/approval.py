"""
Human Approval Node (Agent 4)
=============================
WHAT: Pauses the graph and waits for human approval before generating the report.
HOW:  Uses LangGraph's `interrupt()` to freeze execution. The state is serialized
      to PostgreSQL via the checkpointer. The FastAPI /approve endpoint resumes
      with `Command(resume={"approved": True/False})`.
WHY:  In enterprise settings, AI-generated content MUST be reviewed by humans
      before it's acted upon. This is the most important differentiator from
      toy projects.

FLOW:
  Input:  state["findings"] (presented to human for review)
  Action: interrupt() → state serialized to DB → API exposes findings
  Resume: Command(resume={"approved": True}) → continues to Writer
          Command(resume={"approved": False}) → routes back to Researcher

HOW interrupt() WORKS UNDER THE HOOD:
  1. interrupt() raises an internal GraphInterrupt exception
  2. LangGraph catches this, serializes the ENTIRE state to the checkpointer (PostgreSQL)
  3. The graph execution function returns with status "interrupted"
  4. Later, when Command(resume=value) is called with the same thread_id:
     a. LangGraph loads state from the checkpointer
     b. The interrupt() call RETURNS the value passed to resume
     c. The node continues executing from that exact line
  
  It's like a database-backed `await` — the function suspends, state persists,
  and resumes exactly where it left off, even if the server restarts.

INTERVIEW Q&A:
  Q: How does the interrupt() mechanism work?
  A: interrupt() raises a GraphInterrupt which LangGraph catches and serializes
     the full state to a PostgreSQL checkpointer. The graph pauses. When the user
     calls the /approve endpoint, I invoke the graph with Command(resume=data)
     using the same thread_id. LangGraph loads the persisted state and interrupt()
     RETURNS the resume value. The node continues from that exact line — like a
     coroutine that was awaiting user input.
     
  Q: What if the server restarts while waiting for approval?
  A: The state is persisted in PostgreSQL, not memory. The checkpointer stores
     the full graph state, pending writes, and the position of the interrupt.
     Even after a server restart, resuming with the same thread_id loads the
     correct state and continues. This is why a persistent checkpointer is
     mandatory for production HIL workflows.
     
  Q: Why not just use a simple boolean flag and polling?
  A: That would require you to re-execute the entire graph from the beginning.
     interrupt() preserves the EXACT execution position — all the expensive
     research and extraction work isn't repeated. It's the difference between
     a bookmark and re-reading the entire book.
"""

from langgraph.types import interrupt
from langchain_core.messages import SystemMessage

from app.graph.state import GraphState


async def human_approval_node(state: GraphState) -> dict:
    """
    Human Approval node — pauses graph for human review.
    
    This is where the magic happens:
    1. We summarize findings for the human
    2. Call interrupt() which PAUSES everything
    3. State gets saved to PostgreSQL
    4. FastAPI endpoint later calls Command(resume=...)
    5. interrupt() returns the resume value
    6. We continue from here
    
    Reads: state["findings"], state["confidence_score"]
    Writes: state["human_approved"], state["messages"], state["current_step"]
    """
    findings = state.get("findings", [])
    confidence = state.get("confidence_score", 0.0)

    # Prepare a summary for the human reviewer
    findings_summary = []
    for f in findings[:10]:  # Show top 10 findings
        findings_summary.append(
            f"• [{f.get('confidence', 0):.0%}] {f.get('fact', 'N/A')} (Source: {f.get('source', 'Unknown')})"
        )

    review_data = {
        "message": "Please review the research findings and approve to generate the report.",
        "total_findings": len(findings),
        "average_confidence": round(confidence, 3),
        "topics_covered": list(set(f.get("topic", "Unknown") for f in findings)),
        "top_findings": findings_summary,
        "actions": {
            "approve": "Proceed to report generation",
            "reject": "Send back for more research",
        },
    }

    # ===== THIS IS THE KEY LINE =====
    # interrupt() PAUSES the entire graph here.
    # The state is saved to PostgreSQL.
    # When the user calls /approve, Command(resume=...) makes interrupt() RETURN.
    approval_response = interrupt(review_data)

    # This line only executes AFTER the human responds
    is_approved = approval_response.get("approved", False)
    feedback = approval_response.get("feedback", "")

    return {
        "human_approved": is_approved,
        "human_feedback": feedback if not is_approved else "",
        "current_step": "approved" if is_approved else "rejected",
        "messages": [
            SystemMessage(
                content=f"Human review: {'APPROVED' if is_approved else f'REJECTED — returning to research. Feedback: {feedback}'}"
            ),
        ],
    }
