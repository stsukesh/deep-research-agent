from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from app.config import get_settings


def get_llm(temperature: float = 0.1):
    """
    Get the configured LLM.

    PRIMARY:  NVIDIA NIM (no daily token cap — solves Groq's 100k TPD exhaustion).
    FALLBACK: Groq llama-3.1-8b-instant (ultra-fast when Groq quota is fresh).

    CRITICAL SETTINGS on Groq fallback:
      - max_retries=0  → do not retry internally; propagate exception immediately
                         so LangChain's RunnableWithFallbacks can route to NVIDIA.
      - request_timeout=20 → hard-cap each call; if Groq hangs we surface the
                             error in 20s and NVIDIA takes over.

    WHY NVIDIA FIRST:
      Groq has a 100,000 token/day limit. When exhausted (429), calls would
      block for up to 50 minutes unless max_retries=0 is set. Using NVIDIA as
      primary avoids hitting this limit during normal operation.

    Args:
        temperature: Temperature setting for the model.

    Returns:
        ChatModel instance (either primary or wrapped with fallbacks).
    """
    settings = get_settings()

    # Fast Groq fallback (used when NVIDIA is unavailable or slow)
    # max_retries=0 → fail fast so RunnableWithFallbacks can catch it
    groq_llm = ChatGroq(
        model=settings.GROQ_MODEL,   # llama-3.1-8b-instant — very fast
        temperature=temperature,
        api_key=settings.GROQ_API_KEY,
        max_retries=0,
        request_timeout=20,
    )

    # Primary LLM: NVIDIA NIM (no TPD cap, OpenAI-compatible endpoint)
    if settings.NVIDIA_API_KEY:
        nvidia_llm = ChatOpenAI(
            model=settings.NVIDIA_MODEL,   # meta/llama-3.3-70b-instruct
            temperature=temperature,
            api_key=settings.NVIDIA_API_KEY,
            base_url="https://integrate.api.nvidia.com/v1",
            max_retries=1,
            request_timeout=60,
        )
        # NVIDIA is primary; Groq is the fast fallback
        return nvidia_llm.with_fallbacks([groq_llm])

    # No NVIDIA key → fall back to Groq only
    return groq_llm


def get_fast_llm(temperature: float = 0.1):
    """
    Get the FASTEST possible LLM for text-generation-heavy tasks (Writer, Reviewer).

    PRIMARY:  Groq llama-3.1-8b-instant — 500+ tokens/sec. At 2000 output tokens
              that's ~4 seconds vs ~90 seconds on NVIDIA NIM.
    FALLBACK: NVIDIA NIM — slower but unlimited daily quota.

    Use this for agents that produce large text outputs (Writer, Reviewer).
    Use get_llm() for structured-output tasks (Planner, Extractor) where
    quality matters more than raw generation speed.
    """
    settings = get_settings()

    # Primary: Groq 8b-instant — fastest model available
    groq_fast = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=temperature,
        api_key=settings.GROQ_API_KEY,
        max_retries=0,
        request_timeout=20,
    )

    if settings.NVIDIA_API_KEY:
        nvidia_fallback = ChatOpenAI(
            model=settings.NVIDIA_MODEL,
            temperature=temperature,
            api_key=settings.NVIDIA_API_KEY,
            base_url="https://integrate.api.nvidia.com/v1",
            max_retries=1,
            request_timeout=90,
        )
        return groq_fast.with_fallbacks([nvidia_fallback])

    return groq_fast