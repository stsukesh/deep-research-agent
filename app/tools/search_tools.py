import os
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_tavily import TavilySearch
from app.config import get_settings


def get_search_tools() -> list:
    """
    Returns a list of search tools for the Research Agent.
    
    Tavily: AI-optimized search, returns clean structured results.
    DuckDuckGo: Broad web search, good for general queries.
    """
    settings = get_settings()

    # Ensure the API key is in the environment (Tavily reads from env)
    os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY

    tavily_tool = TavilySearch(
        name="tavily_search",
        description="Search the internet for current information. Best for recent news, market data, and company analysis.",
        max_results=5,
        include_answer=True,
    )

    ddg_tool = DuckDuckGoSearchResults(
        name="duckduckgo_search",
        description="Search the web using DuckDuckGo. Good for general information and diverse sources.",
        max_results=3,
    )

    return [tavily_tool, ddg_tool]