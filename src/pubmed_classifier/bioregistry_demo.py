# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "bioregistry>=0.15.6",
#     "click>=8.5.0",
#     "pandas>=3.1.0",
#     "pubmed-classifier",
# ]
#
# [tool.uv.sources]
# pubmed-classifier = { path = "../../" }
# ///

"""Run a demo on training on Bioregistry papers."""

import bioregistry
import click
import pandas as pd
from bioregistry.constants import CURATED_PAPERS_PATH
from pystow import get_sentence_transformer

from pubmed_classifier import predict_query, train


def _demo() -> None:
    positives = {
        publication.pubmed
        for resource in bioregistry.resources()
        for publication in resource.get_publications()
        if publication.pubmed
    }

    df = pd.read_csv(CURATED_PAPERS_PATH, sep="\t")
    positives.update(df[df["relevant"] == 1].pubmed.map(str))
    negatives = df[df["relevant"] == 0].pubmed.map(str)
    embedder = get_sentence_transformer(device="mps")
    classifiers, _evaluation = train(positives, negatives, embedder=embedder)
    articles, results = predict_query(
        "database OR ontology",
        classifier=classifiers.logistic_regression,
        embedder=embedder,
        retmax=600,
        full=True,
    )
    rows = []
    for article, result in zip(articles, results, strict=False):
        if article.is_review() or article.is_retracted():
            continue
        rows.append((article.pubmed, result, article.title))
    df = pd.DataFrame(rows, columns=["pubmed", "result", "title"])
    df.sort_values("result", ascending=False, inplace=True)
    click.echo(df.to_markdown(index=False))


if __name__ == "__main__":
    _demo()
