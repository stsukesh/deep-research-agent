"""
Arxiv Tool
==========
WHAT: Wraps the Arxiv API as a LangChain tool for academic paper search.
HOW:  ArxivQueryRun uses ArxivAPIWrapper to search and retrieve paper summaries.
WHY:  Academic papers provide the highest-quality technical information — especially
      for AI/ML topics. When researching "Nvidia's AI strategy," Arxiv can surface
      papers on CUDA optimization, transformer acceleration, etc.

INTERVIEW Q&A:
  Q: Why include academic sources in a business research agent?
  A: Technical depth separates a good research report from a superficial one.
     For AI company analysis, understanding the underlying technology (from papers)
     gives context that news articles don't. It also adds credibility — citing
     peer-reviewed papers strengthens the report's authority.
"""

from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper


def get_arxiv_tool() -> ArxivQueryRun:
    """
    Returns an Arxiv academic paper search tool.
    
    top_k_results=3: Get top 3 papers per query
    doc_content_chars_max=4000: Truncate paper summaries
    """
    api_wrapper = ArxivAPIWrapper(
        top_k_results=3,
        doc_content_chars_max=4000,
    )

    return ArxivQueryRun(
        name="arxiv",
        description="Search academic papers on Arxiv. Best for technical topics, AI/ML research, scientific findings, and peer-reviewed sources.",
        api_wrapper=api_wrapper,
    )
