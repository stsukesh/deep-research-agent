"""
Report Writer Agent (Agent 5)
=============================
WHAT: Generates a professional research report from structured findings.
HOW:  Uses get_fast_llm() (Groq 8b-instant, 500+ tok/sec) — the fastest
      available model — to generate a concise 600-900 word markdown report.
WHY:  Report writing is token-generation-heavy. Using the fastest model
      (not the most capable) is the correct trade-off for sub-30s output.

SPEED OPTIMIZATIONS:
  - get_fast_llm(): Groq 8b-instant generates at 500+ tok/sec vs NVIDIA's ~80
  - Target 600-900 words (was 1500-2500) — 3x fewer tokens to generate
  - Top 12 findings only sent to LLM (was all findings) — smaller input
"""

from app.agents.llm_factory import get_fast_llm
from langchain_core.messages import SystemMessage, HumanMessage

from app.graph.state import GraphState


WRITER_SYSTEM_PROMPT = """You are a research report writer. Write concise, professional markdown reports.

Required sections (ALL must be present, each 1-2 paragraphs):
# Executive Summary
# Key Findings  
# Competitive Analysis
# Risks & Opportunities
# Final Assessment
# References

Rules:
- Use inline citations [Source] for every factual claim
- Be specific: use numbers, dates, percentages when available
- Target 600-900 words total — tight and impactful, no padding
- Professional tone, markdown formatting"""


REVISION_PROMPT = """Address this feedback on the report:

{feedback}

Previous report:
{previous_report}

Fix the issues. Keep good content. Be concise."""


async def writer_node(state: GraphState) -> dict:
    """
    Report Writer Agent — uses get_fast_llm() for maximum generation speed.

    Reads: state["findings"], state["query"], state["research_plan"],
           state["review_feedback"] (optional), state["report"] (for revisions)
    Writes: state["report"], state["revision_count"], state["messages"], state["current_step"]
    """
    findings = state.get("findings", [])
    query = state["query"]
    plan = state.get("research_plan", {})
    feedback = state.get("review_feedback", "")
    previous_report = state.get("report", "")
    revision_count = state.get("revision_count", 0)

    # ── SPEED: Groq 8b-instant at 500+ tok/sec ──────────────────────────────
    llm = get_fast_llm(temperature=0.4)

    # Top 12 findings only — keeps input tokens low
    top_findings = findings[:12]
    findings_text = "\n".join(
        f"- [{f.get('topic', 'General')}] {f.get('fact', '')} "
        f"(Source: {f.get('source', 'Unknown')}, Confidence: {f.get('confidence', 0):.0%})"
        for f in top_findings
    )

    if feedback and previous_report:
        human_content = REVISION_PROMPT.format(
            feedback=feedback,
            previous_report=previous_report,
        )
    else:
        human_content = f"""Write a research report for: "{query}"

Scope: {plan.get('scope', 'N/A')}
Objectives: {', '.join(plan.get('objectives', []))}

Findings:
{findings_text}

Write the full report now. Be concise (600-900 words)."""

    messages = [
        SystemMessage(content=WRITER_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    response = await llm.ainvoke(messages)
    report = response.content

    return {
        "report": report,
        "revision_count": revision_count + (1 if feedback else 0),
        "current_step": "report_written",
        "messages": [
            SystemMessage(
                content=f"Report {'revised' if feedback else 'drafted'} — "
                f"revision #{revision_count + (1 if feedback else 0)}"
            ),
        ],
    }

