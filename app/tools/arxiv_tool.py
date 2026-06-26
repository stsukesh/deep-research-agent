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