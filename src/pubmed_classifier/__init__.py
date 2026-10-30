"""Classify documents in PubMed."""

from .api import predict, predict_query, train

__all__ = [
    "predict",
    "predict_query",
    "train",
]
