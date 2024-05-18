"""Query logic for the RAG model."""

import argparse

from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama

from ragit.embedding import get_embedding

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Question: {question}

Context:
---
{context}
---
"""


def main() -> None:
    """Main function to run the CLI."""
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str, debug: bool = False, threshold: float = 300) -> str:
    """Query the RAG model with the given text.

    Args:
        query_text (str): The query text.
        debug (bool, optional): Prints additional information. Defaults to False.
        threshold (float, optional): Threshold of score in RAG query. Defaults to 300.

    Returns:
        str: Returns the formatted response to the query.
    """
    # Prepare the DB.
    embedding_function = get_embedding()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    print(f"Querying with: {query_text}")
    results = [
        (text, score)
        for text, score in db.similarity_search_with_score(query_text, k=15)
        if score < threshold
    ]
    if debug:
        for text, score in results:
            print(f"Score: {score} - {text}")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print("Generating response...")
    model = Ollama(model="gemma:2b")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response


if __name__ == "__main__":
    main()
