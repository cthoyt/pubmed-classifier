"""Prediction workflows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Unpack

import numpy as np
import pubmed_downloader
from pubmed_downloader.client import PubMedSearchKwargs
from pystow import get_sentence_transformer
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from .api import _get_text

__all__ = ["predict", "predict_query"]


def predict_query(
    query: str,
    *,
    embedder: SentenceTransformer | None = None,
    classifier: LogisticRegression,
    return_skipped: bool = False,
    progress: bool = True,
    **search_kwargs: Unpack[PubMedSearchKwargs],
) -> tuple[list[str], np.ndarray] | tuple[list[str], np.ndarray, set[str]]:
    """Classify results from a PubMed query."""
    pubmeds = pubmed_downloader.client.search_with_api(query, **(search_kwargs or {}))
    return predict(
        pubmeds,
        embedder=embedder,
        classifier=classifier,
        return_skipped=return_skipped,
        progress=progress,
    )


def predict(
    pubmeds: Sequence[str],
    *,
    embedder: SentenceTransformer | None = None,
    classifier: LogisticRegression,
    return_skipped: bool = False,
    progress: bool = True,
) -> tuple[list[str], np.ndarray] | tuple[list[str], np.ndarray, set[str]]:
    """Classify documents from PubMed."""
    if embedder is None:
        embedder = get_sentence_transformer()

    texts, pubmeds_rv = [], []
    for article in pubmed_downloader.get_articles(pubmeds, error_strategy="skip"):
        texts.append(_get_text(article))
        pubmeds_rv.append(str(article.pubmed))
    skipped = set(pubmeds) - set(pubmeds_rv)
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=progress)
    results = classifier.predict(embeddings)
    if return_skipped:
        return pubmeds_rv, results, skipped
    else:
        return pubmeds_rv, results
