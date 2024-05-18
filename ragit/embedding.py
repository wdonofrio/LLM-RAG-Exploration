"""This module is used to get the embedding object."""

from langchain_community.embeddings import OllamaEmbeddings


def get_embedding() -> OllamaEmbeddings:
    """Get the embedding object.

    Returns:
        OllamaEmbeddings: Returns OllamaEmbeddings object.
    """
    return OllamaEmbeddings(model="nomic-embed-text")
