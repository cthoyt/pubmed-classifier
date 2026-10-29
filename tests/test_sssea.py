"""Test SSSEA."""

import unittest
from io import StringIO
from textwrap import dedent

from pubmed_classifier.sssea.api import (
    DEFAULT_ANNOTATION_KEY,
    BinaryAnnotation,
    BinaryAnnotationMetadata,
    CategoricalAnnotation,
    CategoricalAnnotationMetadata,
    Metadata,
    Record,
    read,
)


class TestSSSEA(unittest.TestCase):
    """Test case for SSSEA."""

    @staticmethod
    def _read(contents: str, *, metadata: Metadata | None = None) -> list[Record]:
        rv = StringIO(contents)
        rv.seek(0)
        return read(rv, metadata=metadata)

    def _read_one(self, contents: str, *, metadata: Metadata | None = None) -> Record:
        records = self._read(contents, metadata=metadata)
        self.assertEqual(1, len(records))
        record = records[0]
        return record

    def test_raise_on_missing_column(self) -> None:
        """Test raising on missing required column."""
        document = dedent("""\
            entity	curator	date
            pubmed:30380085	orcid:0009-0009-3402-8013	2025-10-07
        """)
        with self.assertRaises(ValueError):
            self._read_one(document)

    def test_raise_on_missing_value(self) -> None:
        """Test raising on missing required key."""
        document = dedent(f"""\
            entity	{DEFAULT_ANNOTATION_KEY}	curator	date
            pubmed:30380085		orcid:0009-0009-3402-8013	2025-10-07
        """)
        with self.assertRaises(ValueError):
            self._read_one(document)

    def test_binary(self) -> None:
        """Test a binary annotation."""
        document = dedent(f"""\
            entity	{DEFAULT_ANNOTATION_KEY}	curator	date
            pubmed:30380085	true	orcid:0009-0009-3402-8013	2025-10-07
        """)
        record = self._read_one(document)
        self.assertEqual("pubmed:30380085", record.reference.curie)
        self.assertEqual(1, len(record.annotations))
        self.assertIn(DEFAULT_ANNOTATION_KEY, record.annotations)
        annotation = record.annotations[DEFAULT_ANNOTATION_KEY]
        self.assertIsInstance(annotation, BinaryAnnotation)
        self.assertTrue(annotation.value)

    def test_binary_custom_column(self) -> None:
        """Test a binary call."""
        key = "key"
        metadata = Metadata(categories=[BinaryAnnotationMetadata(key=key)])
        document = dedent(f"""\
            entity	{key}	curator	date
            pubmed:30380085	true	orcid:0009-0009-3402-8013	2025-10-07
        """)
        record = self._read_one(document, metadata=metadata)
        self.assertEqual("pubmed:30380085", record.reference.curie)
        self.assertEqual(1, len(record.annotations))
        self.assertIn(key, record.annotations)
        annotation = record.annotations[key]
        self.assertIsInstance(annotation, BinaryAnnotation)
        self.assertTrue(annotation.value)

    def test_categorical(self) -> None:
        """Test a binary annotation."""
        metadata = Metadata(
            categories=[CategoricalAnnotationMetadata(key=DEFAULT_ANNOTATION_KEY, nullable=False)]
        )
        document = dedent(f"""\
            entity	{DEFAULT_ANNOTATION_KEY}	curator	date
            pubmed:30380085	a	orcid:0009-0009-3402-8013	2025-10-07
        """)
        record = self._read_one(document, metadata=metadata)
        self.assertEqual("pubmed:30380085", record.reference.curie)
        self.assertEqual(1, len(record.annotations))
        self.assertIn(DEFAULT_ANNOTATION_KEY, record.annotations)
        annotation = record.annotations[DEFAULT_ANNOTATION_KEY]
        self.assertIsInstance(annotation, CategoricalAnnotation)
        self.assertEqual("a", annotation.value)

    def test_multiple(self) -> None:
        """Test a binary annotation."""
        metadata = Metadata(
            categories=[
                BinaryAnnotationMetadata(key="binary_call", nullable=False),
                CategoricalAnnotationMetadata(key="categorical_call", nullable=False),
            ]
        )
        document = dedent("""\
            entity	binary_call categorical_call	curator	date
            pubmed:30380085	true    a	orcid:0009-0009-3402-8013	2025-10-07
        """)
        record = self._read_one(document, metadata=metadata)
        self.assertEqual("pubmed:30380085", record.reference.curie)
        self.assertEqual(2, len(record.annotations))

        self.assertIn("binary_call", record.annotations)
        binary_annotation = record.annotations["binary_call"]
        self.assertIsInstance(binary_annotation, BinaryAnnotation)
        self.assertTrue(binary_annotation.value)

        self.assertIn("categorical_call", record.annotations)
        categorical_annotation = record.annotations["categorical_call"]
        self.assertIsInstance(categorical_annotation, CategoricalAnnotation)
        self.assertEqual("a", categorical_annotation.value)
