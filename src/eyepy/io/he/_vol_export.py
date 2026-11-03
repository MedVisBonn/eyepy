"""Receiving-database metadata for generated VOL exports."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import datetime


@dataclass(frozen=True)
class HeVolExportContext:
    """Identifiers assigned by the HEYEX database that produces an export.

    Importing an E2E into another database may assign new patient, visit
    and scan IDs. They cannot be recovered from the original
    acquisition. Optional fields override source metadata only when
    supplied. These values do not select image registration or establish
    metric fidelity.
    """

    pid: int
    vid: int
    id: str
    reference_id: str | None = None
    patient_id: str | None = None
    visit_id: str | None = None
    prog_id: str | None = None
    exam_time: str | datetime.datetime | None = None
    visit_date: str | datetime.datetime | None = None
    exam_time_ticks: int | None = None
    visit_date_days: float | None = None
    scan_pattern: int | None = None

    def as_dict(self):
        return {key: value for key, value in asdict(self).items() if value is not None}
