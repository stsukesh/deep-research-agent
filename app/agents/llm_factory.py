"""
LLM Factory
===========
WHAT: Centralized factory for instantiating LLM models with built-in fallbacks.
HOW:  Creates a primary ChatGroq instance, and wraps it with a ChatOpenAI fallback
      pointing to NVIDIA NIM if NVIDIA_API_KEY is available.
WHY:  To prevent 429 Rate Limit exceeded errors from blocking agent workflows.

CRITICAL SETTINGS:
  - max_retries=0 on ChatGroq: Disables Groq's internal retry loop. Without this,
    a 429 response causes the Groq client to wait and retry for hours before the
    exception propagates to LangChain's fallback handler.
  - request_timeout=30 on ChatGroq: Hard-caps each API call to 30 seconds. If Groq
    hangs or is slow, this ensures the error surfaces quickly so NVIDIA NIM can
    take over immediately.

INTERVIEW Q&A:
  Q: Why set max_retries=0 instead of letting Groq retry?
  A: Groq's internal retries block the asyncio event loop and bypass the LangChain
     fallback mechanism. By disabling them, any failure (429, 5xx, timeout) is
     immediately raised as an exception, which RunnableWithFallbacks catches and
     routes to the NVIDIA NIM model — giving us fast, transparent failover.

  Q: What is RunnableWithFallbacks?
  A: A LangChain wrapper that runs the primary model, catches any exception, and
     transparently retries using the next model in the fallbacks list. It's the
     standard pattern for multi-provider LLM resilience.
"""

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
