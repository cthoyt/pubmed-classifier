"""Test SSSEA."""

import unittest
from io import StringIO
from textwrap import dedent

from pubmed_classifier.sssea.api import (
    DEFAULT_ANNOTATION_KEY,
    BinaryAnnotation,
    BinaryCategory,
    Metadata,
    Record,
    read,
)


class TestSSSEA(unittest.TestCase):
    """Test case for SSSEA."""

    def _read(self, contents: str, conf: Metadata | None = None) -> list[Record]:
        rv = StringIO(contents)
        rv.seek(0)
        return read(rv, conf)

    def test_binary(self) -> None:
        """Test a binary annotation."""
        document = dedent(f"""\
            entity	{DEFAULT_ANNOTATION_KEY}	curator	date
            pubmed:30380085	true	orcid:0009-0009-3402-8013	2025-10-07
        """)
        records = self._read(document)
        self.assertEqual(1, len(records))
        record = records[0]
        self.assertEqual("pubmed:30380085", record.reference.curie)
        self.assertEqual(1, len(record.annotations))
        self.assertIn(DEFAULT_ANNOTATION_KEY, record.annotations)
        annotation = record.annotations[DEFAULT_ANNOTATION_KEY]
        self.assertIsInstance(annotation, BinaryAnnotation)
        self.assertTrue(annotation.value)

    def test_binary_custom_column(self) -> None:
        """Test a binary call."""
        key = "key"
        metadata = Metadata(categories=[BinaryCategory(key=key)])
        document = dedent(f"""\
            entity	{key}	curator	date
            pubmed:30380085	true	orcid:0009-0009-3402-8013	2025-10-07
        """)
        records = self._read(document, metadata)
        self.assertEqual(1, len(records))
        record = records[0]
        self.assertEqual("pubmed:30380085", record.reference.curie)
        self.assertEqual(1, len(record.annotations))
        self.assertIn(key, record.annotations)
        annotation = record.annotations[key]
        self.assertIsInstance(annotation, BinaryAnnotation)
        self.assertTrue(annotation.value)
