from pathlib import Path

from graph_policy_rag.ingestion.loader import FileSystemDocumentLoader
from graph_policy_rag.chunking.chunker import TextChunker


def main():
    loader = FileSystemDocumentLoader(Path("./documents"))
    docs = loader.load()

    chunker = TextChunker(10, 2)
    print(chunker.chunk(docs[0]))


if __name__ == "__main__":
    main()
