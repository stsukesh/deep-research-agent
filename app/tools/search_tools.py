"""
Search Tools (Tavily + DuckDuckGo)
===================================
WHAT: Wraps Tavily and DuckDuckGo as LangChain-compatible tools for the Research Agent.
HOW:  These are LangChain Tool objects. When the LLM decides to "call a tool," it
      generates a structured function call like {"tool": "tavily_search", "args": {"query": "..."}}.
      LangChain executes the Python function and returns the result to the LLM.
WHY:  The LLM NEVER accesses the internet directly. It delegates to our verified
      tools. This gives us: logging, rate limiting, error handling, and security.

TOOL CALLING FLOW:
  1. LLM sees the tool descriptions in its system prompt
  2. LLM generates: {"tool": "tavily_search", "args": {"query": "Nvidia GPU market share"}}
  3. LangChain intercepts this, calls our Python function
  4. Result is fed back to the LLM for reasoning
  5. LLM can decide to call MORE tools or finalize its response

INTERVIEW Q&A:
  Q: How does tool calling work under the hood?
  A: The tool descriptions (name, args schema, docstring) are injected into the LLM
     prompt. Models like Llama 3.3 have been fine-tuned to output structured tool-call
     JSON. LangChain parses this JSON, executes the matching Python function, and
     appends the result as a ToolMessage. The LLM then reasons over the result.
     
  Q: Why multiple search tools instead of just one?
  A: Different sources excel at different things. Tavily is optimized for AI-ready
     search results (clean, structured). DuckDuckGo provides broader web coverage.
     Using both gives the agent more diverse data to work with, reducing source bias.
"""

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
