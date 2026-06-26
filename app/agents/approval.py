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