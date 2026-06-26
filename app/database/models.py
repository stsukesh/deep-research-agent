"""
Database Models (ORM)
=====================
WHAT: SQLAlchemy ORM models for the app's business data.
HOW:  Two tables — ResearchJob (tracks research pipeline status) and Report
      (stores the final generated report with metrics).
WHY:  LangGraph's checkpointer handles graph state persistence. These tables
      handle BUSINESS logic — job status tracking, report storage, and
      evaluation metrics for the analytics dashboard.

TABLE DESIGN:
  research_jobs:
    - id (UUID): Unique job identifier, returned to the client
    - query (String): The user's original research query
    - status (String): Pipeline status (pending → researching → awaiting_approval → writing → completed)
    - thread_id (String): Maps to LangGraph's thread_id for checkpoint lookup
    - created_at, completed_at: Timestamps for latency tracking

  reports:
    - id (UUID): Unique report identifier
    - job_id (FK → research_jobs.id): Links report to its job
    - report_content (Text): The full markdown report
    - confidence_score, sources_count, revision_count: Quality metrics
    - metrics (JSON): Full evaluation metrics blob

INTERVIEW Q&A:
  Q: Why use UUIDs instead of auto-incrementing integers?
  A: UUIDs are globally unique without coordination — essential in distributed
     systems. If you have multiple app instances creating jobs simultaneously,
     auto-increment IDs would collide. UUIDs also prevent enumeration attacks
     (can't guess /report/2 if you know /report/1 exists).
     
  Q: Why store metrics as JSON instead of separate columns?
  A: Metrics evolve frequently — you might add new ones. JSON columns are
     schema-flexible, so adding a new metric doesn't require a migration.
     For analytics queries, you can use PostgreSQL's JSON operators.
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, String, Float, Integer, Text, DateTime, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class ResearchJob(Base):
    """
    Tracks the lifecycle of a research request.
    
    Status flow: pending → researching → awaiting_approval → writing → completed
                                                ↓ (if rejected)
                                           researching (loop back)
    """
    __tablename__ = "research_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query = Column(String(1000), nullable=False)
    status = Column(String(50), default="pending", index=True)
    thread_id = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationship to reports
    reports = relationship("Report", back_populates="job", lazy="selectin")

    def __repr__(self):
        return f"<ResearchJob {self.id} status={self.status}>"


class Report(Base):
    """
    Stores the final generated report with quality metrics.
    """
    __tablename__ = "reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("research_jobs.id"), nullable=False)
    report_content = Column(Text, nullable=False)
    confidence_score = Column(Float, default=0.0)
    sources_count = Column(Integer, default=0)
    revision_count = Column(Integer, default=0)
    metrics = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship back to job
    job = relationship("ResearchJob", back_populates="reports")

    def __repr__(self):
        return f"<Report {self.id} job={self.job_id} score={self.confidence_score}>"
