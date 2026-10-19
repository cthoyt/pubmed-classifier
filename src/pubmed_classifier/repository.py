"""A repository interface for PubMed classification.

1. Need a query or set of queries
2. Need to decide what kind of featurization
3. Need to specify a directory where data gets stored.

Questions:

1. do I want to have a unified curation + predictions file, or two different ones?

Operations:

1. Incrementally update predictions file with last X days/months/years
2. clean out predictions file based on manual curations
3. retrain / save

Multi-query workflow:

1. If there are multiple possible queries, allow labeling them. Then, keep track of which queries
   each prediction goes with

Exclude rules for predictions:

1. Explicit PubMed IDs
2. explicit journal IDs
3. explicit mesh IDs annotated on articles
3. article types

Tagging:

1. define in configuration the list of allowed "tags", which are an enumeration
   for curation rules or subcategorizations

Web interface:

1. what metadata should get stored in repository?
"""

import datetime

from curies import Reference
from pydantic import BaseModel, Field

#: A mapping from the NLM Catalog ID to the journal name for journals
#: that should be excluded
EXCLUDE_JOURNALS: dict[str, str] = {
    "101680187": "bioRxiv",
    "101767986": "medRxiv",
    # what other preprints can we find? is there an alternate way to get this list?
}


class Curation(BaseModel):
    """Represents a row in the curation table."""

    pubmed: str
    score: float = Field(..., gt=0.0, le=1.0)
    prediction_date: datetime.date
    curator: Reference | None = None
    curation_date: datetime.date | None = None
