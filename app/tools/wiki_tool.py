"""
Wikipedia Tool
==============
WHAT: Wraps the Wikipedia API as a LangChain tool.
HOW:  WikipediaQueryRun uses WikipediaAPIWrapper to search and retrieve articles.
WHY:  Wikipedia provides authoritative background context — company histories,
      technology overviews, industry definitions. It complements the real-time
      data from Tavily/DuckDuckGo.

INTERVIEW Q&A:
  Q: When would an agent choose Wikipedia over a web search?
  A: Wikipedia is best for foundational context — "What is CUDA?", "History of
     Nvidia", "What is a GPU?" Web search is better for recent events and market
     data. The agent's prompt guides this selection, and the tool descriptions
     help the LLM decide which tool to use for each sub-query.
"""

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


def get_wiki_tool() -> WikipediaQueryRun:
    """
    Returns a Wikipedia search tool.
    
    top_k_results=2: Only get top 2 articles (saves tokens)
    doc_content_chars_max=4000: Truncate long articles (saves context window)
    """
    api_wrapper = WikipediaAPIWrapper(
        top_k_results=2,
        doc_content_chars_max=4000,
    )

    return WikipediaQueryRun(
        name="wikipedia",
        description="Search Wikipedia for background information, company histories, technology overviews, and industry definitions.",
        api_wrapper=api_wrapper,
    )
