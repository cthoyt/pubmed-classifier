"""Train, evaluate, and apply PubMed document classifiers."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TypeAlias

import pubmed_downloader
from curies import NamedReference, Prefix
from sssea.api import DEFAULT_ANNOTATION_KEY, BinaryAnnotation, Record
from sssea.learn import (
    BinaryClassifierResult,
    BinaryClassifiers,
    Embedder,
    featurize_article,
    train_binary,
)

__all__ = [
    "BinaryClassifiers",
    "train",
]

logger = logging.getLogger(__name__)

# TODO saving / caching of models


Results: TypeAlias = list[BinaryClassifierResult]
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
