"""
Researcher Agent (Agent 2)
==========================
WHAT: Takes the research plan, iterates through topics, and calls tools to gather data.
HOW:  For each topic, constructs targeted search queries and invokes Tavily, DuckDuckGo,
      Wikipedia, and Arxiv. Results are accumulated in state["search_results"].
WHY:  This is the data-gathering workhorse. It uses ALL available tools to build a
      comprehensive knowledge base before the Extractor refines it.

FLOW:
  Input:  state["research_plan"]["topics"] = ["Revenue", "AI Products", "Competition", ...]
  Output: state["search_results"] = [{"topic": "...", "source": "...", "content": "..."}, ...]

TOOL SELECTION STRATEGY:
  - Business/market topics → Tavily + DuckDuckGo (real-time data)
  - Technical topics → Arxiv (academic papers)
  - Background/history → Wikipedia (authoritative context)
  - All topics get at least one web search for recency

INTERVIEW Q&A:
  Q: How does the agent decide which tool to use?
  A: I use a keyword-based routing strategy. Topics containing words like
     "technical," "algorithm," or "architecture" route to Arxiv. Topics with
     "history" or "background" route to Wikipedia. All topics get a Tavily
     search for current data. This isn't AI-based routing — it's deterministic
     logic that ensures diverse source coverage.
     
  Q: How do you handle API failures and rate limits?
  A: Each tool call is wrapped in try/except with fallback. If Tavily fails,
     we fall back to DuckDuckGo. If all web searches fail for a topic, we
     log the error and continue to the next topic. The system is resilient —
     partial results are better than no results.
"""

import asyncio
from langchain_core.messages import SystemMessage

from app.graph.state import GraphState
from app.tools.search_tools import get_search_tools
from app.tools.wiki_tool import get_wiki_tool
from app.tools.arxiv_tool import get_arxiv_tool


async def _safe_invoke(tool, query: str) -> str:
    """
    Safely invoke a tool with error handling.
    Returns the result string or an error message.
    """
    try:
        result = await asyncio.to_thread(tool.invoke, query)
        if isinstance(result, list):
            # TavilySearchResults returns a list of dicts
            return "\n".join(
                f"- {item.get('content', item.get('snippet', str(item)))}"
                for item in result
            )
        return str(result)
    except Exception as e:
        return f"[Tool Error: {tool.name}] {str(e)}"


async def _research_topic(topic: str, query: str, feedback: str, tools: dict) -> list:
    """
    Research a single topic using all applicable tools.
    Designed to be run concurrently via asyncio.gather().
    """
    topic_results = []
    topic_lower = topic.lower()

    is_technical = any(
        kw in topic_lower
        for kw in ["technical", "algorithm", "architecture", "model", "gpu", "chip", "hardware", "software", "research"]
    )
    is_background = any(
        kw in topic_lower
        for kw in ["history", "background", "overview", "definition", "founded"]
    )

    search_query = f"{query} {topic}"
    if feedback:
        search_query = f"{search_query} {feedback}"

    # === Web Search (always) ===
    web_result = await _safe_invoke(tools["tavily"], search_query)
    topic_results.append({"topic": topic, "source": "tavily_search", "content": web_result})

    # === DuckDuckGo (complementary) ===
    ddg_result = await _safe_invoke(tools["ddg"], search_query)
    topic_results.append({"topic": topic, "source": "duckduckgo_search", "content": ddg_result})

    # === Wikipedia (for background topics) ===
    if is_background or not is_technical:
        wiki_result = await _safe_invoke(tools["wiki"], topic)
        topic_results.append({"topic": topic, "source": "wikipedia", "content": wiki_result})

    # === Arxiv (for technical topics) ===
    if is_technical:
        arxiv_result = await _safe_invoke(tools["arxiv"], topic)
        topic_results.append({"topic": topic, "source": "arxiv", "content": arxiv_result})

    return topic_results


async def researcher_node(state: GraphState) -> dict:
    """
    Researcher Agent node function.

    Researches ALL topics CONCURRENTLY using asyncio.gather() for a major
    speed improvement over sequential processing.

    Reads: state["research_plan"]
    Writes: state["search_results"], state["messages"], state["current_step"]
    """
    plan = state["research_plan"]
    topics = plan.get("topics", [])

    # Initialize all tools once and share across coroutines
    search_tools = get_search_tools()
    tools = {
        "tavily": search_tools[0],
        "ddg":    search_tools[1],
        "wiki":   get_wiki_tool(),
        "arxiv":  get_arxiv_tool(),
    }

    feedback = state.get("human_feedback", "")
    query = state["query"]

    # ── PARALLEL: research all topics at the same time ──────────────────────
    topic_result_lists = await asyncio.gather(
        *[_research_topic(topic, query, feedback, tools) for topic in topics],
        return_exceptions=True,  # don't let one failed topic kill the rest
    )

    all_results = []
    for item in topic_result_lists:
        if isinstance(item, Exception):
            # Log but don't crash — partial results are fine
            all_results.append({
                "topic": "error",
                "source": "error",
                "content": f"[Research error: {item}]",
            })
        else:
            all_results.extend(item)

    return {
        "search_results": all_results,
        "human_feedback": "",  # Clear feedback after consuming it
        "current_step": "research_complete",
        "messages": [
            SystemMessage(
                content=f"Research complete. Gathered {len(all_results)} results across {len(topics)} topics (parallel)."
            ),
        ],
    }

