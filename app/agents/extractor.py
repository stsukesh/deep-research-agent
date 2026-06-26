"""
Information Extraction Agent (Agent 3)
======================================
WHAT: Transforms messy raw search results into clean, structured Finding objects.
HOW:  Uses `with_structured_output(FindingsList)` to force the LLM to extract only
      verifiable facts with source attribution and confidence scores.
WHY:  Raw search results contain ads, irrelevant text, duplicate info. The Extractor
      acts as a data cleaning pipeline — only verified, attributed facts survive.

FLOW:
  Input:  state["search_results"] = [{"topic": "...", "content": "messy text..."}, ...]
  Output: state["findings"] = [{"fact": "...", "source": "...", "confidence": 0.92}, ...]

INTERVIEW Q&A:
  Q: What is information extraction and why is it a separate agent?
  A: Information extraction is the process of transforming unstructured text into
     structured data. It's a separate agent because extraction requires different
     skills than searching. The Researcher is optimized for breadth (finding data),
     the Extractor for precision (filtering noise). Separation of concerns also
     makes each agent testable in isolation.
     
  Q: How do you calculate confidence scores?
  A: The LLM assigns confidence based on: (1) source reliability — Reuters gets
     higher confidence than a random blog, (2) corroboration — facts mentioned
     by multiple sources get boosted, (3) recency — newer data gets slightly
     higher confidence for market analysis. These are LLM-assessed heuristics,
     not algorithmic calculations.
"""

from app.agents.llm_factory import get_llm
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import get_settings
from app.schemas.findings import FindingsList
from app.graph.state import GraphState


EXTRACTOR_SYSTEM_PROMPT = """You are an expert information extraction specialist. Your job is to analyze raw search results and extract verified, factual findings.

For each finding, provide:
1. **fact**: A clear, concise factual statement (one sentence)
2. **source**: The source of this information (e.g., Reuters, Wikipedia, Arxiv paper title)
3. **confidence**: A score from 0.0 to 1.0 based on:
   - 0.9-1.0: From authoritative sources (Reuters, official reports, peer-reviewed papers)
   - 0.7-0.89: From reliable sources (major news outlets, Wikipedia)
   - 0.5-0.69: From secondary sources (blogs, forums, unverified reports)
   - 0.0-0.49: Uncertain or potentially unreliable
4. **topic**: Which research topic this finding relates to

Rules:
- Only extract FACTUAL statements, not opinions or speculation
- Every fact MUST have a source attribution
- Remove duplicate information (keep the version from the most reliable source)
- Aim for 3-5 findings per research topic
- If search results are empty or useless, return fewer findings with appropriate low confidence"""


async def extractor_node(state: GraphState) -> dict:
    """
    Information Extraction Agent node function.
    
    Processes raw search results and extracts structured findings.
    
    Reads: state["search_results"], state["research_plan"]
    Writes: state["findings"], state["confidence_score"], state["messages"], state["current_step"]
    """
    settings = get_settings()
    search_results = state.get("search_results", [])

    if not search_results:
        return {
            "findings": [],
            "confidence_score": 0.0,
            "current_step": "extraction_complete",
            "messages": [SystemMessage(content="No search results to extract from.")],
        }

    # Initialize LLM with structured output and fallback support
    llm = get_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(FindingsList)

    # Format search results for the LLM — truncate aggressively to save tokens
    # Key facts are always in the first ~500 chars; no need to send full content
    MAX_RESULTS = 20   # cap total results sent to LLM
    formatted_results = []
    for result in search_results[:MAX_RESULTS]:
        formatted_results.append(
            f"--- Topic: {result['topic']} | Source: {result['source']} ---\n"
            f"{result['content'][:800]}\n"  # 800 chars is enough for key facts
        )
    results_text = "\n\n".join(formatted_results)

    # Build messages
    messages = [
        SystemMessage(content=EXTRACTOR_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Extract structured findings from these search results:\n\n{results_text}"
        ),
    ]

    # Invoke extraction
    extraction: FindingsList = await structured_llm.ainvoke(messages)

    # Calculate average confidence
    findings_dicts = [f.model_dump() for f in extraction.findings]
    avg_confidence = (
        sum(f.confidence for f in extraction.findings) / len(extraction.findings)
        if extraction.findings
        else 0.0
    )

    return {
        "findings": findings_dicts,
        "confidence_score": round(avg_confidence, 3),
        "current_step": "extraction_complete",
        "messages": [
            SystemMessage(
                content=f"Extracted {len(findings_dicts)} findings with average confidence {avg_confidence:.2f}"
            ),
        ],
    }
