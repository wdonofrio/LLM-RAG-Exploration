"""
This module contains functions to load, transform, and store data.
"""

import re
from pathlib import Path
from typing import List

from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.schema.document import Document


def load_documents(path: Path) -> list[Document]:
    """Load documents from a directory containing files.

    Args:
        path (Path): Directory path to load documents from.

    Returns:
        list[Document]: Returns PyPDFDirectoryLoader object.
    """
    return PyPDFDirectoryLoader(path).load()


def split_documents(
    documents: List[str], max_chunk_size: int = 200, chunk_overlap: int = 50
) -> List[str]:
    """Split documents into smaller chunks, preserving sentence boundaries.

    Args:
        documents (list[str]): List of documents to chunk.
        max_chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        list[str]: Chunked documents.
    """
    chunked_documents = []

    for document in documents:
        sentences_with_newlines = re.split(r"(?<=[\n])", document.page_content)
        sentences = [sent.strip() for sent in sentences_with_newlines if sent.strip()]
        chunks = []
        current_chunk_text = ""
        chunk_size = max_chunk_size

        for sentence in sentences:
            if len(current_chunk_text) + len(sentence) <= chunk_size:
                current_chunk_text += f" {sentence}"
            else:
                # Create a new Document object for the chunk
                chunk_document = Document(
                    page_content=current_chunk_text,
                    metadata=document.metadata.copy(),  # Preserve metadata
                )
                chunks.append(chunk_document)
                current_chunk_text = sentence
                chunk_size = max_chunk_size - chunk_overlap

        if current_chunk_text:
            # Create a new Document object for the remaining chunk
            chunk_document = Document(
                page_content=current_chunk_text,
                metadata=document.metadata.copy(),  # Preserve metadata
            )
            chunks.append(chunk_document)

        chunked_documents.extend(chunks)

    return chunked_documents
