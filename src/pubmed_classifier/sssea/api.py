"""An implementation of SSSLP."""

import datetime
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from curies import NamableReference, Prefix
from pydantic import BaseModel, Field
from pystow.utils import safe_open_dict_reader
from tabulate import tabulate

logger = logging.getLogger(__name__)


class BinaryCategory(BaseModel):
    """Information about how a given category should be parsed."""

    key: str
    reference: NamableReference | None = None


DEFAULT_CATEGORY = "call"


class Conf(BaseModel):
    """Configuration for parsing a literature prioritization TSV document."""

    title: str
    description: str | None = None
    categories: list[BinaryCategory] = Field(
        default_factory=lambda: [BinaryCategory(key=DEFAULT_CATEGORY)]
    )


class BinaryCall(BaseModel):
    """A true/false judgment for a given document."""

    value: bool
    curator: NamableReference | None = None
    date: datetime.date | None = None


class Record(BaseModel):
    """A single curation record."""

    document: NamableReference
    calls: dict[str, BinaryCall]
    abstract: str | None = None


def read(path: str | Path, conf: Conf) -> list[Record]:
    """Read a SSSLP TSV file."""
    return list(_read_iter(path, conf))


def _parse_bool(x: str | None) -> bool | None:
    if not x:
        return None
    if x.lower() in {"yes", "true", "on", "1"}:
        return True
    elif x.lower() in {"no", "false", "off", "0"}:
        return False
    raise ValueError(f"Invalid boolean value: {x}")


def _read_iter(path: str | Path, conf: Conf) -> Iterable[Record]:
    with safe_open_dict_reader(path) as reader:
        # TODO check that the reader's headers include the calls
        for i, record in enumerate(reader, start=2):
            title = record.get("title")
            if document_curie := record.get("document"):
                document = NamableReference.from_curie(document_curie, name=title)
            elif doi := record.get("doi"):
                document = NamableReference(prefix=Prefix("doi"), identifier=doi, name=title)
            elif pubmed := record.get("pubmed"):
                document = NamableReference(prefix=Prefix("pubmed"), identifier=pubmed, name=title)
            else:
                logger.warning("[%d] no document ID found", i)
                continue

            calls = {}
            for category in conf.categories:
                if category.key not in record:
                    continue

                raw_value = record.get(category.key)

                # if extending to have categories that
                # are non-binary, then this would change
                value = _parse_bool(raw_value)
                if value is None:
                    continue

                calls[category.key] = BinaryCall(
                    value=value,
                    curator=NamableReference.from_curie(curator_curie)
                    if (curator_curie := record.get("curator"))
                    else None,
                    date=record.get("date"),
                )
            if not calls:
                logger.warning("[%d] no calls found", i)
                continue

            rr = {
                "document": document,
                "abstract": record.get("abstract"),
                "calls": calls,
            }
            yield Record.model_validate(rr)


def _stratify(
    records: Iterable[Record],
    *,
    category: str | None = None,
) -> tuple[Sequence[str], Sequence[str]]:
    positive = []
    negative = []
    if category is None:
        category = DEFAULT_CATEGORY
    for record in records:
        if record.document.name is None and record.abstract is None:
            continue
        text = " ".join(x for x in (record.document.name, record.abstract) if x is not None)
        if call := record.calls.get(category):
            if call.value is None:
                pass
            elif call.value:
                positive.append(text)
            else:
                negative.append(text)

    return positive, negative


def summarize(records: Iterable[Record], conf: Conf) -> str:
    """Summarize a PubMed classifier."""
    dd: defaultdict[str, Counter[bool | None]] = defaultdict(Counter)
    for record in records:
        for category in conf.categories:
            if call := record.calls.get(category.key):
                dd[category.key][call.value] += 1
            else:
                dd[category.key][None] += 1

    rows = [
        (category.key, dd[category.key][True], dd[category.key][False], dd[category.key][None])
        for category in conf.categories
    ]
    t = tabulate(rows, headers=["Category", "Positive", "Negative", "Missing"], tablefmt="github")
    return t
