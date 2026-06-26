from app.agents.llm_factory import get_fast_llm
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import get_settings
from app.schemas.report import ReviewResult
from app.graph.state import GraphState


REVIEWER_SYSTEM_PROMPT = """You are an expert report quality reviewer. Evaluate the research report against these criteria:

1. **Completeness**: Does it have ALL required sections?
   - Executive Summary, Market Overview, Key Findings, Competitive Analysis,
     Risks & Challenges, Growth Opportunities, Final Assessment, References

2. **Citations**: Does EVERY factual claim have an inline citation [Source Name]?
   - Reports without citations are NOT acceptable

3. **Reasoning**: Are conclusions supported by the findings?
   - No unsupported assertions

4. **Depth**: Is each section substantive (2+ paragraphs)?
   - Sections with just 1-2 sentences need expansion

5. **Consistency**: Does the report contradict itself?

6. **References**: Is there a proper References section listing all sources?

Scoring:
- 7-10: Good to excellent, approve for delivery → status: "approved"
- 5-6.9: Needs targeted improvement → status: "rewrite" with specific, concise feedback
- 0-4.9: Major issues → status: "rewrite" with detailed feedback

Be constructive but efficient. If a report covers all sections with reasonable citations,
approve it even if it could be improved. Only reject if there are MISSING SECTIONS or
NO CITATIONS AT ALL. Speed of delivery matters."""


async def reviewer_node(state: GraphState) -> dict:
    """
    Reviewer Agent node function.
    
    Reviews the report and produces a structured verdict.
    
    Reads: state["report"], state["findings"]
    Writes: state["review_feedback"], state["review_score"],
            state["reviewed_report"], state["messages"], state["current_step"]
    """
    settings = get_settings()
    report = state.get("report", "")
    findings = state.get("findings", [])
    revision_count = state.get("revision_count", 0)

    # Circuit breaker: force approve after MAX_REVISIONS
    if revision_count >= settings.MAX_REVISIONS:
        return {
            "reviewed_report": report,
            "review_feedback": "",
            "review_score": 7.0,
            "current_step": "review_force_approved",
            "messages": [
                SystemMessage(
                    content=f"Force-approved after {revision_count} revisions (circuit breaker)."
                ),
            ],
        }

    # Use fast LLM — reviewer also generates substantial text (feedback)
    llm = get_fast_llm(temperature=0.2)
    structured_llm = llm.with_structured_output(ReviewResult)

    # Build review prompt
    findings_summary = "\n".join(
        f"- {f.get('fact', '')}" for f in findings[:15]
    )

    messages = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(
            content=f"""Review this research report:

=== REPORT START ===
{report}
=== REPORT END ===

Original research findings for fact-checking:
{findings_summary}

This is revision #{revision_count}. Evaluate strictly."""
        ),
    ]

    review: ReviewResult = await structured_llm.ainvoke(messages)

    result = {
        "review_feedback": review.feedback if review.status == "rewrite" else "",
        "review_score": review.score,
        "current_step": f"review_{review.status}",
        "messages": [
            SystemMessage(
                content=f"Review result: {review.status} (score: {review.score}/10). "
                f"{'Feedback: ' + review.feedback if review.status == 'rewrite' else 'Report approved!'}"
            ),
        ],
    }

    if review.status == "approved":
        result["reviewed_report"] = report

    return result