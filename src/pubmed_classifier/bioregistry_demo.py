"""Run a demo on training on Bioregistry papers."""

import bioregistry
import click
import pandas as pd
from bioregistry.constants import CURATED_PAPERS_PATH
from pystow import get_sentence_transformer

from pubmed_classifier.predict import predict_query

from .api import train


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
    classifiers = train(positives, negatives, embedder=embedder)
    pubmeds, results = predict_query(
        "database OR ontology",
        classifier=classifiers.logistic_regression,
        embedder=embedder,
        retmax=600,
    )
    # TODO add title + abstract?
    df = pd.DataFrame({"pubmed": pubmeds, "results": results})
    click.echo(df.to_markdown())


if __name__ == "__main__":
    _demo()
