import sys
sys.path.append('././')
import constants
import numpy as np
import openai
import pandas as pd
import pickle
import tiktoken

COMPLETIONS_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:

    # Currently only the Embedding from openai is supported
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    return result["data"][0]["embedding"]


def compute_doc_embeddings(df: pd.DataFrame) -> dict[int, list[float]]:
    """
    Create an embeddings for each row in the dataframe using the OpenAI Embeddings API.

    Return a dictionary that maps between each embeddings vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.completion) for idx, r in df.iterrows()
    }


def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.

    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))


def order_document_sections_by_query_similarity(query: str, contexts: dict[int, np.array]) -> list[(float, int)]:
    """
    Find the query embeddings for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities

def top_document_sections_by_query_similarity(query: str, contexts: dict[int, np.array]) -> (float, int):
    """
    Find the query embeddings for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the most similar document sections.
    """
    query_embedding = get_embedding(query)

    similarities = []
    for doc_index, doc_embedding in contexts.items():
        similarities.append( vector_similarity(query_embedding, doc_embedding) )

    index_max = np.argmax( similarities )
    document_similarities = ( similarities[index_max], index_max  )

    return document_similarities

def chat_history_to_prompt(question: str, chat_embedding: dict, chat_history: list[str]) -> str:
    most_similar_chat = ""
    if len(chat_embedding) >= 1:
        chat_similarities = top_document_sections_by_query_similarity(question, chat_embedding)
        most_similar_chat = chat_history[chat_similarities[1]]
    return most_similar_chat


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame, model: str = "gpt-3.5-turbo", most_similar_chat: str = "") -> str:
    """
    Fetch relevant
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)

    # print(f"Selected {len(most_relevant_document_sections)} relevant documents")

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
    chosen_sections_similarities = []

    for similarity, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        document_section = df.loc[section_index]
        document_section_tokens = len(encoding.encode(df.loc[section_index]["completion"]))

        chosen_sections.append(SEPARATOR + df.loc[section_index]["completion"].replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
        chosen_sections_similarities.append(similarity)

        chosen_sections_len += document_section_tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

    # Useful diagnostic information
    # print(f"Selected {len(chosen_sections)} document sections and their similarity index:")
    # print("\n".join(i + ": " + str(sim) for i, sim in zip(chosen_sections_indexes, chosen_sections_similarities)))
    # print( chosen_sections )

    if most_similar_chat != "":
        most_similar_chat = SEPARATOR + most_similar_chat

    prefix_Q = "Q: "

    return "".join(chosen_sections) + most_similar_chat + "\n\n " + constants.prefix_Q + question