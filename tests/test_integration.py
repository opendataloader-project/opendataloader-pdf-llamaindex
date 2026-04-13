"""Integration tests for OpenDataLoader PDF Reader.

These tests require Java 11+ and actual PDF files.
They are skipped automatically when Java is not available.
"""

import json

import pytest
from llama_index.core.schema import Document

from llama_index.readers.opendataloader_pdf import OpenDataLoaderPDFReader

from .conftest import java_available

pytestmark = pytest.mark.skipif(
    not java_available(), reason="Java 11+ required for integration tests"
)


class TestIntegrationBasic:
    """Basic loading tests with all 4 formats."""

    @pytest.mark.parametrize("fmt", ["text", "markdown", "json", "html"])
    def test_load_single_pdf(self, sample_pdf, fmt) -> None:
        reader = OpenDataLoaderPDFReader(format=fmt, split_pages=False)
        docs = list(reader.load_data(file_path=sample_pdf))
        assert len(docs) >= 1
        assert isinstance(docs[0], Document)
        assert docs[0].text
        assert docs[0].metadata["source"] == "lorem.pdf"
        assert docs[0].metadata["format"] == fmt

    def test_load_multiple_pdfs(self, sample_pdfs) -> None:
        reader = OpenDataLoaderPDFReader(split_pages=False)
        docs = list(reader.load_data(file_path=sample_pdfs))
        assert len(docs) >= 2


class TestIntegrationSplitPages:
    """Test per-page splitting across formats."""

    def test_split_pages_text(self, multi_page_pdf) -> None:
        reader = OpenDataLoaderPDFReader(format="text", split_pages=True)
        docs = list(reader.load_data(file_path=multi_page_pdf))
        assert len(docs) > 1
        assert all("page" in doc.metadata for doc in docs)

    def test_split_pages_markdown(self, multi_page_pdf) -> None:
        reader = OpenDataLoaderPDFReader(format="markdown", split_pages=True)
        docs = list(reader.load_data(file_path=multi_page_pdf))
        assert len(docs) > 1

    def test_split_pages_json(self, multi_page_pdf) -> None:
        reader = OpenDataLoaderPDFReader(format="json", split_pages=True)
        docs = list(reader.load_data(file_path=multi_page_pdf))
        assert len(docs) > 1
        # Each doc should be valid JSON.
        for doc in docs:
            data = json.loads(doc.text)
            assert "kids" in data
            assert "page number" in data

    def test_split_pages_html(self, multi_page_pdf) -> None:
        reader = OpenDataLoaderPDFReader(format="html", split_pages=True)
        docs = list(reader.load_data(file_path=multi_page_pdf))
        assert len(docs) > 1

    def test_no_split(self, multi_page_pdf) -> None:
        reader = OpenDataLoaderPDFReader(format="text", split_pages=False)
        docs = list(reader.load_data(file_path=multi_page_pdf))
        assert len(docs) == 1
        assert "page" not in docs[0].metadata


class TestIntegrationOptions:
    """Test various extraction options."""

    def test_sanitize(self, sample_pdf) -> None:
        reader = OpenDataLoaderPDFReader(sanitize=True, split_pages=False)
        docs = list(reader.load_data(file_path=sample_pdf))
        assert len(docs) >= 1

    def test_pages_selection(self, multi_page_pdf) -> None:
        reader = OpenDataLoaderPDFReader(pages="1", split_pages=False)
        docs = list(reader.load_data(file_path=multi_page_pdf))
        assert len(docs) == 1

    def test_use_struct_tree(self, sample_pdf) -> None:
        reader = OpenDataLoaderPDFReader(use_struct_tree=True, split_pages=False)
        docs = list(reader.load_data(file_path=sample_pdf))
        assert len(docs) >= 1

    def test_keep_line_breaks(self, sample_pdf) -> None:
        reader = OpenDataLoaderPDFReader(keep_line_breaks=True, split_pages=False)
        docs = list(reader.load_data(file_path=sample_pdf))
        assert len(docs) >= 1


class TestIntegrationPathHandling:
    """Test different path input types."""

    def test_string_path(self, sample_pdf) -> None:
        reader = OpenDataLoaderPDFReader(split_pages=False)
        docs = list(reader.load_data(file_path=str(sample_pdf)))
        assert len(docs) >= 1

    def test_path_object(self, sample_pdf) -> None:
        reader = OpenDataLoaderPDFReader(split_pages=False)
        docs = list(reader.load_data(file_path=sample_pdf))
        assert len(docs) >= 1

    def test_list_of_paths(self, sample_pdfs) -> None:
        reader = OpenDataLoaderPDFReader(split_pages=False)
        docs = list(reader.load_data(file_path=sample_pdfs))
        assert len(docs) >= 2


class TestIntegrationExtraInfo:
    """Test extra_info merging in integration context."""

    def test_extra_info_preserved(self, sample_pdf) -> None:
        reader = OpenDataLoaderPDFReader(split_pages=False)
        docs = list(
            reader.load_data(
                file_path=sample_pdf,
                extra_info={"category": "test"},
            )
        )
        assert docs[0].metadata["category"] == "test"
        assert docs[0].metadata["source"] == "lorem.pdf"
