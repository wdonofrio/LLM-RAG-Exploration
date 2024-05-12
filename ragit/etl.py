"""
This module contains functions to load, transform, and store data.
"""

from pathlib import Path

from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(path: Path) -> list[Document]:
    """Load documents from a directory containing files.

    Args:
        path (Path): Directory path to load documents from.

    Returns:
        list[Document]: Returns PyPDFDirectoryLoader object.
    """
    return PyPDFDirectoryLoader(path).load()


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks.

    Args:
        documents (list[Document]): List of documents to chunk.

    Returns:
        list[Document]: Chunked documents.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=150, chunk_overlap=50, length_function=len, is_separator_regex=False
    ).split_documents(documents)
