"""
FastAPI Routes
==============
WHAT: REST API endpoints for the research agent system.
HOW:  FastAPI router with async endpoints. Graph uses AsyncPostgresSaver (Neon)
      so all state is durable — jobs survive server restarts.
WHY:  The API decouples the client from the agent execution. Clients submit
      research requests, poll for status, approve findings, and retrieve reports.

ENDPOINTS:
  POST /research       → Start a new research job (async)
  GET  /status/{id}    → Check job status + findings (for approval)
  POST /approve/{id}   → Approve or reject findings (HIL)
  GET  /report/{id}    → Get the final report
  GET  /jobs           → List all research jobs
  GET  /metrics/{id}   → Get evaluation metrics
  GET  /health         → Health check
"""

import asyncio
import time
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langgraph.types import Command
from sqlalchemy import select, update
from datetime import datetime

from app.graph.graph_builder import build_graph
from app.evaluations.metrics import collect_metrics
from app.database.connection import get_session_factory, init_db
from app.database.models import ResearchJob, Report
from app.config import get_settings

# Clear cached settings so fresh .env is always read on startup
get_settings.cache_clear()

router = APIRouter()

# Shared graph instance (set during lifespan startup)
graph = None

# Keep strong references to background tasks to prevent GC mid-execution
background_tasks: set[asyncio.Task] = set()


async def startup(checkpointer=None):
    """Initialize DB tables and build the graph with the provided checkpointer."""
    global graph
    await init_db()                          # create ORM tables (research_jobs, reports)
    graph = build_graph(checkpointer)        # build and compile graph using checkpointer


# ===== Request/Response Models =====

class ResearchRequest(BaseModel):
    query: str

class ApprovalRequest(BaseModel):
    approved: bool = True
    feedback: str = ""

class JobResponse(BaseModel):
    job_id: str
    query: str
    status: str
    thread_id: str
    message: str


# ===== DB helpers =====

async def _create_job(job_id: str, query: str, thread_id: str):
    factory = get_session_factory()
    async with factory() as session:
        job = ResearchJob(
            id=uuid.UUID(job_id),
            query=query,
            status="pending",
            thread_id=thread_id,
        )
        session.add(job)
        await session.commit()


async def _update_job_status(job_id: str, status: str, completed: bool = False):
    try:
        val = uuid.UUID(job_id)
    except ValueError:
        return
    factory = get_session_factory()
    async with factory() as session:
        values = {"status": status}
        if completed:
            values["completed_at"] = datetime.utcnow()
        await session.execute(
            update(ResearchJob).where(ResearchJob.id == val).values(**values)
        )
        await session.commit()


async def _get_job(job_id: str):
    try:
        val = uuid.UUID(job_id)
    except ValueError:
        return None
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(ResearchJob).where(ResearchJob.id == val)
        )
        return result.scalar_one_or_none()


async def _save_report(job_id: str, result: dict, start_time: float):
    try:
        val = uuid.UUID(job_id)
    except ValueError:
        return
    report_content = result.get("reviewed_report") or result.get("report", "No report generated.")
    metrics_data = collect_metrics(result, start_time)
    factory = get_session_factory()
    async with factory() as session:
        report = Report(
            job_id=val,
            report_content=report_content,
            confidence_score=result.get("confidence_score", 0.0),
            sources_count=len(result.get("findings", [])),
            revision_count=result.get("revision_count", 0),
            metrics=metrics_data.to_dict(),
        )
        session.add(report)
        await session.commit()


async def _get_report(job_id: str):
    try:
        val = uuid.UUID(job_id)
    except ValueError:
        return None
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(Report).where(Report.job_id == val)
        )
        return result.scalar_one_or_none()


# ===== Background Tasks =====

async def run_research_pipeline(job_id: str, query: str, thread_id: str):
    start_time = time.time()
    try:
        await _update_job_status(job_id, "researching")
        config = {"configurable": {"thread_id": thread_id}}
        result = await graph.ainvoke({"query": query}, config=config)

        state = await graph.aget_state(config)
        if state.next:
            await _update_job_status(job_id, "awaiting_approval")
            return

        await _update_job_status(job_id, "completed", completed=True)
        await _save_report(job_id, result, start_time)

    except Exception as e:
        await _update_job_status(job_id, "failed")
        # Store error in a lightweight way by updating status field
        factory = get_session_factory()
        async with factory() as session:
            await session.execute(
                update(ResearchJob)
                .where(ResearchJob.id == uuid.UUID(job_id))
                .values(status=f"failed: {str(e)[:200]}")
            )
            await session.commit()


async def resume_research_pipeline(job_id: str, request: ApprovalRequest, config: dict, start_time: float):
    try:
        result = await graph.ainvoke(
            Command(resume={"approved": request.approved, "feedback": request.feedback}),
            config=config,
        )
        state = await graph.aget_state(config)
        if state.next:
            await _update_job_status(job_id, "awaiting_approval")
            return

        await _update_job_status(job_id, "completed", completed=True)
        await _save_report(job_id, result, start_time)

    except Exception as e:
        await _update_job_status(job_id, "failed")


# ===== Endpoints =====

@router.post("/research", response_model=JobResponse)
async def start_research(request: ResearchRequest):
    job_id = str(uuid.uuid4())
    thread_id = f"research_{job_id}"

    await _create_job(job_id, request.query, thread_id)

    task = asyncio.create_task(run_research_pipeline(job_id, request.query, thread_id))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)

    return JobResponse(
        job_id=job_id,
        query=request.query,
        status="pending",
        thread_id=thread_id,
        message="Research started. Poll GET /status/{job_id} for updates.",
    )


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    job = await _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {
        "job_id": str(job.id),
        "query": job.query,
        "status": job.status,
        "thread_id": job.thread_id,
        "created_at": job.created_at.isoformat() if job.created_at else None,
    }

    if job.status in ("researching", "awaiting_approval"):
        config = {"configurable": {"thread_id": job.thread_id}}
        try:
            state = await graph.aget_state(config)
            if state.tasks and any(
                hasattr(t, "interrupts") and t.interrupts for t in state.tasks
            ):
                await _update_job_status(job_id, "awaiting_approval")
                response["status"] = "awaiting_approval"
                for t in state.tasks:
                    if hasattr(t, "interrupts") and t.interrupts:
                        response["review_data"] = t.interrupts[0].value
                        break
        except Exception:
            pass

    report = await _get_report(job_id)
    if report:
        response["has_report"] = True

    return response


@router.post("/approve/{job_id}")
async def approve_research(job_id: str, request: ApprovalRequest):
    job = await _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    new_status = "writing" if request.approved else "researching"
    await _update_job_status(job_id, new_status)

    config = {"configurable": {"thread_id": job.thread_id}}
    start_time = job.created_at.timestamp() if job.created_at else time.time()

    task = asyncio.create_task(resume_research_pipeline(job_id, request, config, start_time))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)

    return {
        "job_id": job_id,
        "approved": request.approved,
        "status": new_status,
        "message": "Processing approval... Check status for updates.",
    }


@router.get("/report/{job_id}")
async def get_report(job_id: str):
    job = await _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    report = await _get_report(job_id)
    if not report:
        raise HTTPException(
            status_code=400,
            detail=f"Report not ready. Current status: {job.status}",
        )

    return {
        "job_id": job_id,
        "query": job.query,
        "report": report.report_content,
        "confidence_score": report.confidence_score,
        "revision_count": report.revision_count,
        "findings_count": report.sources_count,
    }


@router.get("/jobs")
async def list_jobs():
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(select(ResearchJob).order_by(ResearchJob.created_at.desc()).limit(50))
        jobs = result.scalars().all()
    return {
        "jobs": [
            {
                "job_id": str(j.id),
                "query": j.query,
                "status": j.status,
                "thread_id": j.thread_id,
                "created_at": j.created_at.isoformat() if j.created_at else None,
            }
            for j in jobs
        ],
        "total": len(jobs),
    }


@router.get("/metrics/{job_id}")
async def get_metrics(job_id: str):
    job = await _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    report = await _get_report(job_id)
    if not report:
        raise HTTPException(status_code=400, detail="Metrics not yet available.")

    return {
        "job_id": job_id,
        "query": job.query,
        "metrics": report.metrics,
    }


@router.get("/debug/{job_id}")
async def debug_job(job_id: str):
    job = await _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    config = {"configurable": {"thread_id": job.thread_id}}
    history = []
    try:
        async for state in graph.aget_state_history(config):
            history.append({
                "checkpoint_id": state.config.get("configurable", {}).get("checkpoint_id"),
                "next": state.next,
                "values": {
                    "current_step": state.values.get("current_step"),
                    "human_approved": state.values.get("human_approved"),
                    "revision_count": state.values.get("revision_count"),
                },
            })
    except Exception as e:
        return {"error": str(e)}

    return {"job_status": job.status, "history": history}


@router.get("/health")
async def health_check():
    factory = get_session_factory()
    db_ok = False
    try:
        async with factory() as session:
            await session.execute(select(ResearchJob).limit(1))
        db_ok = True
    except Exception:
        pass

    return {
        "status": "healthy" if db_ok else "degraded",
        "service": "enterprise-research-agent",
        "database": "neon-postgres" if db_ok else "unreachable",
        "active_jobs": len([t for t in background_tasks if not t.done()]),
    }
