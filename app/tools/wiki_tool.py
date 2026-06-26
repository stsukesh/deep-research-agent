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