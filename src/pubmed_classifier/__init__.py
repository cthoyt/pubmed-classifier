"""Classify documents in PubMed."""

from .api import train
from .predict import predict, predict_query

__all__ = [
    "predict",
    "predict_query",
    "train",
]
