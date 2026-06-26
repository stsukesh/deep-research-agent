"""
Tests for Individual Agent Nodes
=================================
Tests each agent in isolation by providing mock state inputs
and verifying the output state updates are correct.

INTERVIEW Q&A:
  Q: How do you test AI agents?
  A: I test at two levels: (1) Unit tests mock the LLM and verify state
     transformations — does the Planner return a valid research_plan dict?
     Does the Extractor return findings with confidence scores? (2) Integration
     tests run the full graph with a real LLM but mocked tools. The key insight
     is that agent nodes are just async functions that take state and return
     state updates — they're fully testable.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.graph.state import GraphState


@pytest.mark.asyncio
async def test_planner_node_returns_research_plan():
    """Planner should return a research_plan dict with topics."""
    from app.agents.planner import planner_node

    mock_plan = MagicMock()
    mock_plan.model_dump.return_value = {
        "topics": ["Revenue", "AI Products", "Competition"],
        "objectives": ["Analyze market position"],
        "scope": "Nvidia AI business analysis",
    }
    mock_plan.topics = ["Revenue", "AI Products", "Competition"]

    with patch("app.agents.planner.get_llm") as MockGetLLM:
        # Mock chain: get_llm() → llm.with_structured_output() → structured_llm → structured_llm.ainvoke()
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_plan)

        mock_instance = MagicMock()
        mock_instance.with_structured_output.return_value = mock_structured_llm
        MockGetLLM.return_value = mock_instance

        state: GraphState = {
            "query": "Analyze Nvidia AI strategy",
            "research_plan": {},
            "search_results": [],
            "findings": [],
            "human_approved": False,
            "report": "",
            "reviewed_report": "",
            "review_feedback": "",
            "review_score": 0.0,
            "confidence_score": 0.0,
            "revision_count": 0,
            "current_step": "",
            "messages": [],
            "metrics": {},
        }

        result = await planner_node(state)

        # Verify output
        assert "research_plan" in result
        assert "topics" in result["research_plan"]
        assert len(result["research_plan"]["topics"]) > 0
        assert result["current_step"] == "planning_complete"


@pytest.mark.asyncio
async def test_extractor_node_returns_findings():
    """Extractor should return structured findings with confidence scores."""
    from app.agents.extractor import extractor_node

    mock_findings = MagicMock()
    mock_finding = MagicMock()
    mock_finding.confidence = 0.85
    mock_finding.model_dump.return_value = {
        "fact": "Nvidia controls 80% of AI GPU market",
        "source": "Reuters",
        "confidence": 0.85,
        "topic": "Market Share",
    }
    mock_findings.findings = [mock_finding]

    with patch("app.agents.extractor.get_llm") as MockGetLLM:
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_findings)

        mock_instance = MagicMock()
        mock_instance.with_structured_output.return_value = mock_structured_llm
        MockGetLLM.return_value = mock_instance

        state: GraphState = {
            "query": "Analyze Nvidia",
            "research_plan": {"topics": ["Market Share"]},
            "search_results": [
                {"topic": "Market Share", "source": "tavily", "content": "Nvidia leads GPU market..."}
            ],
            "findings": [],
            "human_approved": False,
            "report": "",
            "reviewed_report": "",
            "review_feedback": "",
            "review_score": 0.0,
            "confidence_score": 0.0,
            "revision_count": 0,
            "current_step": "",
            "messages": [],
            "metrics": {},
        }

        result = await extractor_node(state)

        assert "findings" in result
        assert len(result["findings"]) > 0
        assert "confidence_score" in result
        assert result["confidence_score"] > 0


@pytest.mark.asyncio
async def test_reviewer_circuit_breaker():
    """Reviewer should force-approve after MAX_REVISIONS."""
    from app.agents.reviewer import reviewer_node

    with patch("app.agents.reviewer.get_settings") as mock_settings:
        mock_settings.return_value.MAX_REVISIONS = 3
        mock_settings.return_value.GROQ_MODEL = "test"
        mock_settings.return_value.GROQ_API_KEY = "test"

        state: GraphState = {
            "query": "Test",
            "research_plan": {},
            "search_results": [],
            "findings": [],
            "human_approved": True,
            "report": "# Test Report\nSome content.",
            "reviewed_report": "",
            "review_feedback": "",
            "review_score": 0.0,
            "confidence_score": 0.5,
            "revision_count": 3,  # At max revisions
            "current_step": "",
            "messages": [],
            "metrics": {},
        }

        result = await reviewer_node(state)

        # Should force-approve
        assert result["reviewed_report"] == "# Test Report\nSome content."
        assert "force_approved" in result["current_step"]


@pytest.mark.asyncio
async def test_researcher_appends_feedback():
    """Researcher should append human feedback to web search queries and then clear it."""
    from app.agents.researcher import researcher_node

    # Mock get_search_tools to return mock tool objects
    mock_tavily = MagicMock()
    mock_tavily.invoke = MagicMock(return_value="tavily result")
    mock_tavily.name = "tavily_search"
    
    mock_ddg = MagicMock()
    mock_ddg.invoke = MagicMock(return_value="ddg result")
    mock_ddg.name = "duckduckgo_search"

    mock_wiki = MagicMock()
    mock_wiki.invoke = MagicMock(return_value="wiki result")
    mock_wiki.name = "wikipedia"

    mock_arxiv = MagicMock()
    mock_arxiv.invoke = MagicMock(return_value="arxiv result")
    mock_arxiv.name = "arxiv"

    with patch("app.agents.researcher.get_search_tools", return_value=[mock_tavily, mock_ddg]), \
         patch("app.agents.researcher.get_wiki_tool", return_value=mock_wiki), \
         patch("app.agents.researcher.get_arxiv_tool", return_value=mock_arxiv):

        state: GraphState = {
            "query": "Analyze Nvidia",
            "research_plan": {"topics": ["Revenue"]},
            "search_results": [],
            "findings": [],
            "human_approved": False,
            "human_feedback": "focus on Q3 earnings",
            "report": "",
            "reviewed_report": "",
            "review_feedback": "",
            "review_score": 0.0,
            "confidence_score": 0.0,
            "revision_count": 0,
            "current_step": "",
            "messages": [],
            "metrics": {},
        }

        result = await researcher_node(state)

        # Check queries invoked
        mock_tavily.invoke.assert_called_once_with("Analyze Nvidia Revenue focus on Q3 earnings")
        mock_ddg.invoke.assert_called_once_with("Analyze Nvidia Revenue focus on Q3 earnings")

        # Verify state updates: feedback cleared
        assert result["human_feedback"] == ""
        assert len(result["search_results"]) > 0


@pytest.mark.asyncio
async def test_approval_node_saves_feedback():
    """Approval node should return human_feedback state updates when interrupted and resumed."""
    from app.agents.approval import human_approval_node

    with patch("app.agents.approval.interrupt", return_value={"approved": False, "feedback": "Need more competitor data"}) as mock_interrupt:
        state: GraphState = {
            "query": "Analyze Nvidia",
            "research_plan": {},
            "search_results": [],
            "findings": [{"fact": "GPU leader", "confidence": 0.9, "topic": "Revenue", "source": "Reuters"}],
            "human_approved": False,
            "human_feedback": "",
            "report": "",
            "reviewed_report": "",
            "review_feedback": "",
            "review_score": 0.0,
            "confidence_score": 0.9,
            "revision_count": 0,
            "current_step": "",
            "messages": [],
            "metrics": {},
        }

        result = await human_approval_node(state)

        assert result["human_approved"] is False
        assert result["human_feedback"] == "Need more competitor data"
        assert result["current_step"] == "rejected"
        assert "Feedback: Need more competitor data" in result["messages"][0].content


def test_llm_factory_returns_fallback_runnable():
    """get_llm should return a RunnableWithFallbacks when NVIDIA_API_KEY is configured."""
    from app.agents.llm_factory import get_llm
    from langchain_core.runnables import RunnableWithFallbacks
    
    with patch("app.agents.llm_factory.get_settings") as mock_settings:
        mock_settings.return_value.GROQ_MODEL = "llama-3.3-70b-versatile"
        mock_settings.return_value.GROQ_API_KEY = "groq-key"
        mock_settings.return_value.NVIDIA_API_KEY = "nvidia-key"
        mock_settings.return_value.NVIDIA_MODEL = "meta/llama-3.3-70b-instruct"
        
        llm = get_llm(temperature=0.1)
        
        assert isinstance(llm, RunnableWithFallbacks)
        assert llm.runnable.model_name == "meta/llama-3.3-70b-instruct"
        assert llm.runnable.openai_api_key.get_secret_value() == "nvidia-key"
        assert llm.runnable.openai_api_base == "https://integrate.api.nvidia.com/v1"
        assert len(llm.fallbacks) == 1
        assert llm.fallbacks[0].model == "llama-3.3-70b-versatile"


def test_llm_factory_returns_primary_only():
    """get_llm should return only the primary ChatGroq model when NVIDIA_API_KEY is not configured."""
    from app.agents.llm_factory import get_llm
    from langchain_groq import ChatGroq
    
    with patch("app.agents.llm_factory.get_settings") as mock_settings:
        mock_settings.return_value.GROQ_MODEL = "llama-3.3-70b-versatile"
        mock_settings.return_value.GROQ_API_KEY = "groq-key"
        mock_settings.return_value.NVIDIA_API_KEY = ""
        
        llm = get_llm(temperature=0.1)
        
        assert isinstance(llm, ChatGroq)


