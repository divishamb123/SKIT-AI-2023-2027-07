import uuid
import enum
from datetime import datetime
from sqlalchemy import (
    String,
    Text,
    BigInteger,
    SmallInteger,
    DateTime,
    Enum,
    ForeignKey,
    CheckConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from shared.db.base import Base


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModalityType(str, enum.Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint(
            "(input_type = 'text' AND input_text IS NOT NULL AND input_object_key IS NULL) OR "
            "(input_type = 'file' AND input_object_key IS NOT NULL AND input_text IS NULL)",
            name="chk_job_input",
        ),
        CheckConstraint("priority BETWEEN 1 AND 10", name="chk_job_priority"),
        CheckConstraint("input_type IN ('text', 'file')", name="chk_job_input_type"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[JobStatus] = mapped_column(
        Enum(
            JobStatus,
            name="job_status",
            native_enum=True,
            values_callable=lambda obj: [e.value for e in obj],
        ),
        nullable=False,
        default=JobStatus.QUEUED,
    )
    modality: Mapped[ModalityType] = mapped_column(
        Enum(
            ModalityType,
            name="modality_type",
            native_enum=True,
            values_callable=lambda obj: [e.value for e in obj],
        ),
        nullable=False,
    )
    input_type: Mapped[str] = mapped_column(String(10), nullable=False)
    input_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_object_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    file_name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    file_mime_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    file_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)

    priority: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=5)
    retry_count: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    max_retries: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=3)
    queue_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)

    queued_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="jobs")
    reports: Mapped[list["Report"]] = relationship(
        "Report", back_populates="job", cascade="all, delete-orphan"
    )
