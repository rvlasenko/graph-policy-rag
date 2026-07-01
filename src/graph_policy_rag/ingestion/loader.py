from pathlib import Path
import hashlib

from graph_policy_rag.domain.document import Document
from graph_policy_rag.exceptions import DataDirectoryNotFoundError


class FileSystemDocumentLoader:
    def __init__(self, documents_dir: Path) -> None:
        self.documents_dir = documents_dir

    def load(self) -> list[Document]:
        if not self.documents_dir.exists():
            raise DataDirectoryNotFoundError(
                f"Data directory not found: {self.documents_dir}"
            )

        if not self.documents_dir.is_dir():
            raise DataDirectoryNotFoundError(
                f"Provided path is not a directory: {self.documents_dir}"
            )

        documents: list[Document] = []
        for md_file in self.documents_dir.rglob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            relative_path = md_file.relative_to(self.documents_dir)

            document_id = _make_hash(str(relative_path))
            version = _make_hash(content)

            documents.append(
                Document(
                    id=document_id,
                    title=md_file.stem,
                    content=content,
                    source_uri=md_file.resolve().as_uri(),
                    metadata={},
                    version=version,
                )
            )

        return documents


def _make_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
