"""Prediction workflows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import pubmed_downloader
from pubmed_downloader import Article
from pubmed_downloader.client import PubMedSearchKwargs
from sssea.learn import Embedder
from typing_extensions import Unpack

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from sklearn.linear_model import LogisticRegression

from sssea.learn import predict_binary

__all__ = ["predict", "predict_query"]


# docstr-coverage:excused `overload`
@overload
def predict_query(
    query: str,
    *,
    embedder: Embedder | None = ...,
    classifier: LogisticRegression,
    progress: bool = ...,
    full: Literal[False] = False,
    **search_kwargs: Unpack[PubMedSearchKwargs],
) -> tuple[list[str], NDArray[np.float64]]: ...


# docstr-coverage:excused `overload`
@overload
def predict_query(
    query: str,
    *,
    embedder: Embedder | None = ...,
    classifier: LogisticRegression,
    progress: bool = ...,
    full: Literal[True] = True,
    **search_kwargs: Unpack[PubMedSearchKwargs],
) -> tuple[list[Article], NDArray[np.float64]]: ...


def predict_query(
    query: str,
    *,
    embedder: Embedder | None = None,
    classifier: LogisticRegression,
    progress: bool = True,
    full: bool = False,
    **search_kwargs: Unpack[PubMedSearchKwargs],
) -> tuple[list[str], NDArray[np.float64]] | tuple[list[Article], NDArray[np.float64]]:
    """Classify results from a PubMed query."""
    pubmeds = pubmed_downloader.client.search_with_api(query, **(search_kwargs or {}))
    return predict(  # type:ignore[call-overload,no-any-return]
        pubmeds,
        embedder=embedder,
        classifier=classifier,
        progress=progress,
        full=full,
    )


# docstr-coverage:excused `overload`
@overload
def predict(
    pubmeds: Sequence[str],
    *,
    embedder: Embedder | None,
    classifier: LogisticRegression,
    progress: bool,
    full: Literal[False],
) -> tuple[list[str], NDArray[np.float64]]: ...


# docstr-coverage:excused `overload`
@overload
def predict(
    pubmeds: Sequence[str],
    *,
    embedder: Embedder | None,
    classifier: LogisticRegression,
    progress: bool,
    full: Literal[True],
) -> tuple[list[Article], NDArray[np.float64]]: ...


def predict(
    pubmeds: Sequence[str],
    *,
    embedder: Embedder | None = None,
    classifier: LogisticRegression,
    progress: bool = True,
    full: bool = False,
) -> tuple[list[str], NDArray[np.float64]] | tuple[list[Article], NDArray[np.float64]]:
    """Classify documents from PubMed."""
    if embedder is None:
        from pystow import get_sentence_transformer

        embedder = get_sentence_transformer()

    articles = list(pubmed_downloader.get_articles(pubmeds, error_strategy="skip"))
    texts, pubmeds_rv = [], []
    for article in articles:
        texts.append(_get_text(article))
        pubmeds_rv.append(str(article.pubmed))
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=progress)
    results = predict_binary(classifier, embeddings)
    if full:
        return articles, results
    else:
        return pubmeds_rv, results


def _get_text(article: Article) -> str:
    return article.title + " " + article.get_abstract()
