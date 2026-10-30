"""Train, evaluate, and apply PubMed document classifiers."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import pubmed_downloader
import pystow
from curies import NamedReference, Prefix
from pubmed_downloader import Article
from pubmed_downloader.client import PubMedSearchKwargs
from sssea.api import DEFAULT_ANNOTATION_KEY, BinaryAnnotation, Record
from sssea.learn import (
    BinaryClassifierResult,
    BinaryClassifiers,
    Embedder,
    embed_text,
    featurize_article,
    predict_binary,
    train_binary,
)
from typing_extensions import Unpack

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from sklearn.linear_model import LogisticRegression

__all__ = [
    "predict",
    "predict_query",
    "train",
]

logger = logging.getLogger(__name__)

# TODO saving / caching of models


PUBMED = Prefix("pubmed")


def train(
    positive_pubmed_ids: Iterable[str],
    negative_pubmed_ids: Iterable[str],
    *,
    embedder: Embedder | None = None,
) -> tuple[BinaryClassifiers, list[BinaryClassifierResult]]:
    """Train a PubMed classifier based on positive and negative identifier sets."""
    return train_binary(
        _iter_records(positive_pubmed_ids, negative_pubmed_ids, key=DEFAULT_ANNOTATION_KEY),
        embedder=embedder,
        featurizer=featurize_article,
        key=DEFAULT_ANNOTATION_KEY,
    )


def _iter_records(
    positive_pubmed_ids: Iterable[str],
    negative_pubmed_ids: Iterable[str],
    key: str,
) -> Iterable[Record]:
    for pubmed_ids, indicator in [
        (positive_pubmed_ids, True),
        (negative_pubmed_ids, False),
    ]:
        for article in pubmed_downloader.get_articles(pubmed_ids, error_strategy="skip"):
            yield Record(
                reference=NamedReference(
                    prefix=PUBMED, identifier=str(article.pubmed), name=article.title
                ),
                annotations={key: BinaryAnnotation(value=indicator)},
                features={"abstract": article.get_abstract()},
            )


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
        embedder = pystow.get_sentence_transformer()

    articles = list(pubmed_downloader.get_articles(pubmeds, error_strategy="skip"))
    texts, pubmeds_rv = [], []
    for article in articles:
        texts.append(_get_text(article))
        pubmeds_rv.append(str(article.pubmed))

    embeddings = embed_text(embedder, texts, show_progress_bar=progress)
    results = predict_binary(classifier, embeddings)
    if full:
        return articles, results
    else:
        return pubmeds_rv, results


def _get_text(article: Article) -> str:
    return article.title + " " + article.get_abstract()
