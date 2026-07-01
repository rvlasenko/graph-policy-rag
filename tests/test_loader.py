import pytest

from graph_policy_rag.ingestion.loader import FileSystemDocumentLoader
from graph_policy_rag.exceptions import DataDirectoryNotFoundError


@pytest.fixture
def documents_dir(tmp_path):
    path = tmp_path / "documents"
    path.mkdir()
    return path


def test_loads_markdown_documents(documents_dir):
    file_path = documents_dir / "seller_onboarding.md"
    file_path.write_text("# Seller onboarding\nRules here.", encoding="utf-8")

    loader = FileSystemDocumentLoader(documents_dir)

    documents = loader.load()

    assert len(documents) == 1
    assert documents[0].title == "seller_onboarding"
    assert documents[0].content == "# Seller onboarding\nRules here."
    assert documents[0].source_uri == file_path.resolve().as_uri()
    assert documents[0].metadata == {}
    assert documents[0].version is not None


def test_ignores_non_markdown_files(documents_dir):
    (documents_dir / "seller.md").write_text("Seller rules", encoding="utf-8")
    (documents_dir / "notes.txt").write_text("Internal notes", encoding="utf-8")
    (documents_dir / "image.png").write_text("fake image", encoding="utf-8")

    loader = FileSystemDocumentLoader(documents_dir)

    documents = loader.load()

    assert len(documents) == 1
    assert documents[0].title == "seller"
    assert documents[0].content == "Seller rules"


def test_raises_if_directory_does_not_exist(tmp_path):
    missing_dir = tmp_path / "missing"

    loader = FileSystemDocumentLoader(missing_dir)

    with pytest.raises(DataDirectoryNotFoundError):
        loader.load()


def test_keeps_document_id_stable(documents_dir):
    file_path = documents_dir / "seller_onboarding.md"
    file_path.write_text("# Seller onboarding\nRules here.", encoding="utf-8")

    loader = FileSystemDocumentLoader(documents_dir)

    first_document = loader.load()[0]
    document_id = first_document.id

    file_path.write_text(
        "# Seller onboarding\nRules here.\nNew changes.", encoding="utf-8"
    )

    updated_document = loader.load()[0]

    assert updated_document.id == document_id


def test_changes_version_when_content_changes(documents_dir):
    file_path = documents_dir / "seller_onboarding.md"
    file_path.write_text("# Seller onboarding\nRules here.", encoding="utf-8")

    loader = FileSystemDocumentLoader(documents_dir)

    first_document = loader.load()[0]
    version = first_document.version

    file_path.write_text(
        "# Seller onboarding\nRules here.\nNew changes.", encoding="utf-8"
    )

    updated_document = loader.load()[0]

    assert updated_document.version != version
