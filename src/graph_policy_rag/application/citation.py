from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    document_id: str
    source_uri: str
