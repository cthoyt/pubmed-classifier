"""Learning from SSSEA."""

from collections.abc import Iterable
from typing import NamedTuple

import click
import numpy as np
from curies import NamableReference
from numpy._typing import NDArray
from sentence_transformers import SentenceTransformer
from tabulate import tabulate

from pubmed_classifier.api import Classifiers, Results, _embed, _predict, _train_helper
from pubmed_classifier.sssea.api import (
    BinaryAnnotationMetadata,
    Features,
    Featurizer,
    Metadata,
    Record,
    read,
    stratify_featurize_binary,
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

    positives, negatives = stratify_featurize_binary(
        records, category=category, featurizer=featurizer
    )
    if not positives or not negatives:
        raise ValueError("missing one of positives/negatives. check that category is set properly")

    positive_embeddings, positive_indicator = _embed(positives, embedder, True)
    negative_embeddings, negative_indicator = _embed(negatives, embedder, False)

    x = np.vstack((positive_embeddings, negative_embeddings))
    y = np.hstack((positive_indicator, negative_indicator))
    return PreparedData(x, y)


def _featurize_abstract(reference: NamableReference, features: Features) -> str | None:
    abstract = features.get("abstract")
    if reference.name is None and abstract is None:
        return None
    text = " ".join(x for x in (reference.name, abstract) if x is not None)
    return text


@click.command()
def _demo() -> None:
    import pystow

    conf = Metadata(
        title="test",
        description="test",
        categories=[
            BinaryAnnotationMetadata(key="enzyme paper"),
            BinaryAnnotationMetadata(key="PET depolymerization"),
            BinaryAnnotationMetadata(key="PEF depolymerization"),
            BinaryAnnotationMetadata(key="PEF polymerization"),
            BinaryAnnotationMetadata(key="PET polymerization"),
        ],
    )

    records = read("/Users/cthoyt/dev/catalaix/app/sources/citavi/papers.tsv", metadata=conf)
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
