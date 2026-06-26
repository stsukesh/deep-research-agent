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
    Research a single topic using all applicable tools concurrently.
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

    # Prepare concurrent tasks
    tasks = []
    task_keys = []

    # 1. Tavily Search (always)
    tasks.append(_safe_invoke(tools["tavily"], search_query))
    task_keys.append("tavily_search")

    # 2. DuckDuckGo Search (always)
    tasks.append(_safe_invoke(tools["ddg"], search_query))
    task_keys.append("duckduckgo_search")

    # 3. Wikipedia (for background / non-technical topics)
    run_wiki = is_background or not is_technical
    if run_wiki:
        tasks.append(_safe_invoke(tools["wiki"], topic))
        task_keys.append("wikipedia")

    # 4. Arxiv (for technical topics)
    run_arxiv = is_technical
    if run_arxiv:
        tasks.append(_safe_invoke(tools["arxiv"], topic))
        task_keys.append("arxiv")

    # Run searches concurrently
    results = await asyncio.gather(*tasks)

    # Map results back
    for key, result in zip(task_keys, results):
        topic_results.append({"topic": topic, "source": key, "content": result})

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