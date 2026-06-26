import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture(autouse=True)
async def app_lifespan():
    async with app.router.lifespan_context(app):
        yield


@pytest.mark.asyncio
async def test_health_check():
    """Health endpoint should return 200 with status healthy."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "enterprise-research-agent"


@pytest.mark.asyncio
async def test_list_jobs_empty():
    """Jobs endpoint should return empty list initially."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data


@pytest.mark.asyncio
async def test_start_research():
    """POST /research should return a job_id and status pending."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/research",
            json={"query": "Test research query"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert data["query"] == "Test research query"


@pytest.mark.asyncio
async def test_status_not_found():
    """GET /status for non-existent job should return 404."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/status/nonexistent-id")
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_report_not_ready():
    """GET /report before completion should return 400."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First create a job
        res = await client.post(
            "/api/research",
            json={"query": "Test"},
        )
        job_id = res.json()["job_id"]

        # Try to get report immediately (not ready)
        response = await client.get(f"/api/report/{job_id}")
        assert response.status_code == 400


@pytest.mark.asyncio
async def test_serve_ui():
    """Root endpoint should serve the web UI HTML."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")