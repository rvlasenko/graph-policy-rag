from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    content: str
    source_uri: str
    metadata: dict[str, str]
    version: str | None = None
    updated_at: datetime | None = None
