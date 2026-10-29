"""An implementation of SSSEA."""

import datetime
import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import Annotated, Any, Generic, Literal, TypeAlias, TypeVar

from curies import NamableReference, Prefix
from pydantic import BaseModel, Field
from pystow.utils import Source, safe_open_dict_reader
from tabulate import tabulate

__all__ = [
    "DEFAULT_ANNOTATION_KEY",
    "BinaryAnnotation",
    "BinaryAnnotationMetadata",
    "CategoricalAnnotation",
    "CategoricalAnnotationMetadata",
    "Features",
    "Featurizer",
    "Metadata",
    "Record",
    "read",
    "stratify_featurize_binary",
]

logger = logging.getLogger(__name__)

AnnotationValueType = TypeVar("AnnotationValueType")

AnnotationType: TypeAlias = Literal["binary", "categorical"]

#: A dictionary of features
Features: TypeAlias = dict[str, Any]

#: A function that consumes a features dictionary and produces a feature string
Featurizer: TypeAlias = Callable[[NamableReference, Features], str | None]


class _AnnotationMetadata(BaseModel):
    annotation_type: AnnotationType
    key: str
    predicate: Annotated[
        NamableReference | None,
        Field(default=None, description="the predicate connecting the entity to the annotation"),
    ] = None
    nullable: bool = True


class BinaryAnnotationMetadata(_AnnotationMetadata):
    """Information about how a given category should be parsed."""

    annotation_type: Literal["binary"] = "binary"


class CategoricalAnnotationMetadata(_AnnotationMetadata):
    """Information about how a given category should be parsed."""

    annotation_type: Literal["categorical"] = "categorical"
    allowed: list[str] | None = None


AnnotationMetadata = Annotated[
    BinaryAnnotationMetadata | CategoricalAnnotationMetadata, Field(discriminator="annotation_type")
]

DEFAULT_ANNOTATION_KEY = "call"


def _default_category() -> list[AnnotationMetadata]:
    return [BinaryAnnotationMetadata(key=DEFAULT_ANNOTATION_KEY, nullable=False)]


class Metadata(BaseModel):
    """Configuration for parsing a literature prioritization TSV document."""

    title: str | None = None
    description: str | None = None
    license: NamableReference | None = None
    creators: list[NamableReference] | None = None
    categories: list[AnnotationMetadata] = Field(default_factory=_default_category)


class _Annotation(BaseModel, Generic[AnnotationValueType]):
    annotation_type: AnnotationType
    value: AnnotationValueType
    curator: NamableReference | None = None
    date: datetime.date | None = None


class BinaryAnnotation(_Annotation[bool]):
    """A binary judgment."""

    annotation_type: Literal["binary"] = "binary"
    value: bool


class CategoricalAnnotation(_Annotation[str]):
    """A categorical judgment."""

    annotation_type: Literal["categorical"] = "categorical"
    value: str


#: An annotation, either binary or categorical
Annotation = Annotated[
    BinaryAnnotation | CategoricalAnnotation, Field(discriminator="annotation_type")
]


class Record(BaseModel):
    """A single curation record."""

    reference: NamableReference
    annotations: dict[str, Annotation]
    features: Features


def read(path: Source, *, metadata: Metadata | None = None) -> list[Record]:
    """Read a SSSLP TSV file."""
    return list(_read_iter(path, metadata=metadata))


def _parse_bool(x: str) -> bool:
    if x.lower() in {"yes", "true", "on", "1"}:
        return True
    elif x.lower() in {"no", "false", "off", "0"}:
        return False
    raise ValueError(f"Invalid boolean value: {x}")


DOI = Prefix("doi")
PUBMED = Prefix("pubmed")


def _read_iter(path: Source, *, metadata: Metadata | None = None) -> Iterable[Record]:  # noqa:C901
    if metadata is None:
        metadata = Metadata()

    # check that metadata doesn't have duplicate keys
    counter = Counter(category.key for category in metadata.categories)
    duplicates = Counter({k: count for k, count in counter.items() if count > 1})
    if duplicates:
        raise ValueError(f"duplicate categories defined in metadata: {duplicates}")

    with safe_open_dict_reader(path) as reader:
        if missing_columns := sorted(
            category.key
            for category in metadata.categories
            if not category.nullable
            and (not reader.fieldnames or category.key not in reader.fieldnames)
        ):
            raise ValueError(
                f"missing columns for required annotations: {missing_columns}, "
                f"got {reader.fieldnames}"
            )

        # If name and title in reader columns, raise
        # TODO check that the reader's headers include the defined calls and features
        for i, record in enumerate(reader, start=2):
            name = record.get("name") or record.get("title")
            if entity_curie := record.get("entity"):
                reference = NamableReference.from_curie(entity_curie, name=name)
            elif doi := record.get("doi"):
                reference = NamableReference(prefix=DOI, identifier=doi, name=name)
            elif pubmed_id := record.get("pubmed"):
                reference = NamableReference(prefix=PUBMED, identifier=pubmed_id, name=name)
            else:
                logger.warning("[%d] no entity found", i)
                continue

            features: dict[str, Any] = {}

            annotations: dict[str, Annotation] = {}
            for category in metadata.categories:
                value = record.get(category.key)
                if not value:  # cath both explicit none and empty string
                    if category.nullable:
                        continue
                    else:
                        raise ValueError(f"required to have category: {category.key}")

                curator = (
                    NamableReference.from_curie(curator_curie)
                    if (curator_curie := record.get("curator"))
                    else None
                )

                match category:
                    case BinaryAnnotationMetadata():
                        annotations[category.key] = BinaryAnnotation(
                            value=_parse_bool(value), curator=curator, date=record.get("date")
                        )
                    case CategoricalAnnotationMetadata():
                        if category.allowed and value not in category.allowed:
                            raise ValueError
                        annotations[category.key] = CategoricalAnnotation(
                            value=value, curator=curator, date=record.get("date")
                        )

            if not annotations:
                logger.warning("[%d] no calls found", i)
                continue

            yield Record(reference=reference, annotations=annotations, features=features)


def stratify_featurize_binary(
    records: Iterable[Record],
    *,
    featurizer: Featurizer,
    category: str | None = None,
) -> tuple[Sequence[str], Sequence[str]]:
    """Stratify records based on a binary category."""
    positive = []
    negative = []
    if category is None:
        category = DEFAULT_ANNOTATION_KEY
    for record in records:
        text = featurizer(record.reference, record.features)
        if text is None:
            continue

        match record.annotations.get(category):
            case None:
                continue
            case BinaryAnnotation() as annotation:
                if annotation.value:
                    positive.append(text)
                else:
                    negative.append(text)
            case CategoricalAnnotation():
                # this could be implemented by passing
                # a specific one of the tags to consider
                # as either positive or negative, then
                # considering it as a one-vs-all scenario
                raise NotImplementedError

    return positive, negative


def summarize(records: Iterable[Record], conf: Metadata) -> str:
    """Summarize a PubMed classifier."""
    dd: defaultdict[str, Counter[str | bool | None]] = defaultdict(Counter)
    for record in records:
        for category in conf.categories:
            if call := record.annotations.get(category.key):
                dd[category.key][call.value] += 1
            else:
                dd[category.key][None] += 1

    rows = [
        (category.key, dd[category.key][True], dd[category.key][False], dd[category.key][None])
        for category in conf.categories
    ]
    t = tabulate(rows, headers=["Category", "Positive", "Negative", "Missing"], tablefmt="github")
    return t
