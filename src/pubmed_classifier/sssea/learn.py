"""Learning from SSSEA."""

from collections.abc import Iterable
from typing import NamedTuple

import click
import numpy as np
from numpy._typing import NDArray
from sentence_transformers import SentenceTransformer
from tabulate import tabulate

from pubmed_classifier.api import Classifiers, Results, _embed, _predict, _train_helper
from pubmed_classifier.sssea.api import (
    BinaryCategory,
    Featurizer,
    Metadata,
    Record,
    _featurize_abstract,
    _stratify,
    read,
    summarize,
)

__all__ = ["prepare", "train"]


def train(
    records: Iterable[Record],
    *,
    featurizer: Featurizer,
    category: str | None = None,
    embedder: SentenceTransformer | None = None,
) -> tuple[Classifiers, Results]:
    """Train a PubMed classifier based on positive and negative identifier sets."""
    x, y = prepare(records, category=category, embedder=embedder, featurizer=featurizer)
    return _train_helper(x, y)


class PreparedData(NamedTuple):
    """A pair of inputs (X) and corresponding labels (y)."""

    x: NDArray[np.float64]
    y: NDArray[np.bool]


def prepare(
    records: Iterable[Record],
    *,
    featurizer: Featurizer,
    category: str | None = None,
    embedder: SentenceTransformer | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.bool]]:
    """Prepare SSSEA records for learning."""
    if embedder is None:
        import pystow

        embedder = pystow.get_sentence_transformer()

    positives, negatives = _stratify(records, category=category, featurizer=featurizer)
    if not positives or not negatives:
        raise ValueError("missing one of positives/negatives. check that category is set properly")

    positive_embeddings, positive_indicator = _embed(positives, embedder, True)
    negative_embeddings, negative_indicator = _embed(negatives, embedder, False)

    x = np.vstack((positive_embeddings, negative_embeddings))
    y = np.hstack((positive_indicator, negative_indicator))
    return PreparedData(x, y)


@click.command()
def _demo() -> None:
    import pystow

    conf = Metadata(
        title="test",
        description="test",
        categories=[
            BinaryCategory(key="enzyme paper"),
            BinaryCategory(key="PET depolymerization"),
            BinaryCategory(key="PEF depolymerization"),
            BinaryCategory(key="PEF polymerization"),
            BinaryCategory(key="PET polymerization"),
        ],
    )
    records = read("/Users/cthoyt/dev/catalaix/app/sources/citavi/papers.tsv", conf)
    click.echo(summarize(records, conf))
    embedder = pystow.get_sentence_transformer()
    _classifiers, results = train(
        records, category="PEF depolymerization", embedder=embedder, featurizer=_featurize_abstract
    )
    click.echo(
        tabulate([(n, v) for n, _, v in results], headers=["model", "ROC-AUC"], tablefmt="github")
    )

    import pandas as pd

    df = pd.read_csv(
        "/Users/cthoyt/dev/catalaix/app/prioritization/collated.tsv", sep="\t", dtype=str
    )

    vecs = embedder.encode(df["title"], convert_to_numpy=True)
    df["PEF depolymerization"] = _predict(_classifiers.logistic_regression, vecs)
    df = df.sort_values("PEF depolymerization", ascending=False)
    df.to_csv("pef-depolymerization.tsv", index=False, sep="\t")


if __name__ == "__main__":
    _demo()
