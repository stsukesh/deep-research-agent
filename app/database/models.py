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