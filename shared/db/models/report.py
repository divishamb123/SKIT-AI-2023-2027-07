import uuid
import enum
from datetime import datetime
from sqlalchemy import (
    String,
    Text,
    BigInteger,
    Integer,
    DateTime,
    Enum,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from shared.db.base import Base


class ReportType(str, enum.Enum):
    PDF = "pdf"
    JSON = "json"
    HTML = "html"


class ReportStatus(str, enum.Enum):
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class Report(Base):
    __tablename__ = "reports"
    __table_args__ = (
        UniqueConstraint("job_id", "report_type", name="uq_reports_job_report_type"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    report_type: Mapped[ReportType] = mapped_column(
        Enum(ReportType, name="report_type", native_enum=True), nullable=False
    )
    status: Mapped[ReportStatus] = mapped_column(
        Enum(ReportStatus, name="report_status", native_enum=True),
        nullable=False,
        default=ReportStatus.PENDING,
    )
    minio_object_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    file_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    signed_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    signed_url_expires: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    generation_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    generated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    # Relationships
    job: Mapped["Job"] = relationship("Job", back_populates="reports")
