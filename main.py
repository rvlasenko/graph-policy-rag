from pathlib import Path

from graph_policy_rag.ingestion.loader import FileSystemDocumentLoader


def main():
    loader = FileSystemDocumentLoader(Path("./documents"))
    docs = loader.load()
    print(docs)


if __name__ == "__main__":
    main()
