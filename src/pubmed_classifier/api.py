"""Train, evaluate, and apply PubMed document classifiers."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, NamedTuple, cast

import numpy as np
import pubmed_downloader
from numpy.typing import NDArray
from pystow import get_sentence_transformer
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from typing_extensions import Self

__all__ = [
    "Classifiers",
    "train",
]

logger = logging.getLogger(__name__)

# TODO saving / caching of models


class Classifiers(NamedTuple):
    """A tuple containing a variety of classifiers."""

    random_forest: RandomForestClassifier
    logistic_regression: LogisticRegression
    decision_tree: DecisionTreeClassifier
    linear_svc: LinearSVC
    rbf_svc: SVC

    @classmethod
    def make(cls) -> Self:
        """Construct an ensemble of classifiers."""
        return cls(
            random_forest=RandomForestClassifier(),
            logistic_regression=LogisticRegression(),
            decision_tree=DecisionTreeClassifier(),
            linear_svc=LinearSVC(),
            rbf_svc=SVC(kernel="rbf", probability=True),
        )


def _prepare(
    positives: Iterable[str],
    negatives: Iterable[str],
    *,
    embedder: SentenceTransformer | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.bool]]:
    if embedder is None:
        embedder = get_sentence_transformer()

    positive_embeddings, positive_labels = _embed(positives, embedder, True)
    negative_embeddings, negative_labels = _embed(negatives, embedder, False)

    x = np.vstack((positive_embeddings, negative_embeddings))
    y = np.hstack((positive_labels, negative_labels))
    return x, y


def train(
    positives: Iterable[str],
    negatives: Iterable[str],
    *,
    embedder: SentenceTransformer | None = None,
) -> Classifiers:
    """Train a PubMed classifier based on positive and negative identifier sets."""
    x, y = _prepare(positives, negatives, embedder=embedder)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, random_state=42, shuffle=True
    )

    classifiers = Classifiers.make()
    results = []
    for key, classifier in zip(Classifiers._fields, classifiers, strict=False):
        classifier.fit(x_train, y_train)
        roc_auc = roc_auc_score(y_test, _predict(classifier, x_test))
        logger.info("%s ROC-AUC: %.2f", key, roc_auc)
        results.append((key, classifier, roc_auc))

    return classifiers


def _predict(classifier: Any, x: NDArray[np.float64] | NDArray[np.str_]) -> NDArray[np.float64]:
    if hasattr(classifier, "predict_proba"):
        return cast(NDArray[np.float64], classifier.predict_proba(x)[:, 1])
    elif hasattr(classifier, "decision_function"):
        return cast(NDArray[np.float64], classifier.decision_function(x))
    else:
        raise TypeError(f"classifier {classifier} neither has predict_proba nor decision function")


def _embed(
    pubmeds: Iterable[str],
    embedder: SentenceTransformer | TfidfVectorizer,
    positive: bool,
) -> tuple[NDArray[np.float64], NDArray[np.bool]]:
    articles = pubmed_downloader.get_articles(pubmeds, error_strategy="skip")
    texts = [_get_text(article) for article in articles]
    embeddings = _embedddd(embedder, texts)
    b = [positive] * embeddings.shape[0]
    rr = np.array(b, dtype=np.bool)
    return embeddings, rr


def _embedddd(
    embedder: SentenceTransformer | TfidfVectorizer, texts: list[str], **kwargs: Any
) -> NDArray[np.float64]:
    if isinstance(embedder, SentenceTransformer):
        return cast(NDArray[np.float64], embedder.encode(texts, convert_to_numpy=True, **kwargs))
    elif isinstance(embedder, TfidfVectorizer):
        return cast(NDArray[np.float64], embedder.transform(texts, **kwargs))
    else:
        raise TypeError(f"embedder type {type(embedder)} is not supported")


def _get_text(article: pubmed_downloader.Article) -> str:
    return article.title + " " + article.get_abstract()
