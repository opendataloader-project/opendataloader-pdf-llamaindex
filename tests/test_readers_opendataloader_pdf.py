"""Unit tests for OpenDataLoader PDF Reader."""

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from llama_index.readers.opendataloader_pdf import OpenDataLoaderPDFReader

# Save original before any monkeypatching.
_original_path_exists = Path.exists


@pytest.fixture(autouse=True)
def _bypass_input_validation(monkeypatch):
    """Bypass file-existence and Java checks in unit tests.

    Tests that verify validation behaviour override these explicitly.
    Only dummy PDF paths are faked; all other Path.exists calls delegate
    to the real implementation.
    """
    monkeypatch.setattr(
        "llama_index.readers.opendataloader_pdf.base._java_available",
        lambda: True,
    )
    monkeypatch.setattr("llama_index.readers.opendataloader_pdf.base._java_found", None)

    def _fake_exists(self):
        if self.suffix == ".pdf":
            return True
        return _original_path_exists(self)

    monkeypatch.setattr("pathlib.Path.exists", _fake_exists)


# ---------------------------------------------------------------------------
# TestInit: Verify default and custom parameter values
# ---------------------------------------------------------------------------
class TestInit:
    """Test reader instantiation and Pydantic field defaults."""

    def test_defaults(self) -> None:
        reader = OpenDataLoaderPDFReader()
        assert reader.format == "text"
        assert reader.quiet is False
        assert reader.split_pages is True
        assert reader.content_safety_off is None
        assert reader.password is None
        assert reader.keep_line_breaks is False
        assert reader.replace_invalid_chars is None
        assert reader.use_struct_tree is False
        assert reader.table_method is None
        assert reader.reading_order is None
        assert reader.image_output == "off"
        assert reader.image_format is None
        assert reader.image_dir is None
        assert reader.sanitize is False
        assert reader.pages is None
        assert reader.include_header_footer is False
        assert reader.detect_strikethrough is False
        assert reader.hybrid is None
        assert reader.hybrid_mode is None
        assert reader.hybrid_url is None
        assert reader.hybrid_timeout is None
        assert reader.hybrid_fallback is False

    def test_custom_values(self) -> None:
        reader = OpenDataLoaderPDFReader(
            format="json",
            quiet=True,
            split_pages=False,
            sanitize=True,
            hybrid="docling-fast",
            pages="1-3",
        )
        assert reader.format == "json"
        assert reader.quiet is True
        assert reader.split_pages is False
        assert reader.sanitize is True
        assert reader.hybrid == "docling-fast"
        assert reader.pages == "1-3"


# ---------------------------------------------------------------------------
# TestFormatValidation: Invalid format raises ValueError
# ---------------------------------------------------------------------------
class TestFormatValidation:
    """Test that invalid format raises ValueError."""

    def test_invalid_format(self) -> None:
        reader = OpenDataLoaderPDFReader(format="invalid")
        with pytest.raises(ValueError, match="Invalid format"):
            list(reader.load_data(file_path="dummy.pdf"))

    def test_format_case_insensitive(self) -> None:
        reader = OpenDataLoaderPDFReader(format="TEXT")
        # Should not raise on format validation (would fail on convert)
        with patch("opendataloader_pdf.convert"):
            with patch(
                "llama_index.readers.opendataloader_pdf.base.tempfile.mkdtemp",
                return_value="/tmp/test",
            ):
                with patch(
                    "llama_index.readers.opendataloader_pdf.base.Path.glob",
                    return_value=[],
                ):
                    with patch("llama_index.readers.opendataloader_pdf.base.shutil.rmtree"):
                        list(reader.lazy_load_data(file_path="dummy.pdf"))


# ---------------------------------------------------------------------------
# TestConvertCall: Verify convert() is called with correct kwargs
# ---------------------------------------------------------------------------
class TestConvertCall:
    """Test that opendataloader_pdf.convert() receives correct arguments."""

    @patch("llama_index.readers.opendataloader_pdf.base.shutil.rmtree")
    @patch("llama_index.readers.opendataloader_pdf.base.Path.glob", return_value=[])
    @patch("llama_index.readers.opendataloader_pdf.base.tempfile.mkdtemp")
    @patch("opendataloader_pdf.convert")
    def test_default_kwargs(
        self, mock_convert: MagicMock, mock_mkdtemp: MagicMock, *_: MagicMock
    ) -> None:
        mock_mkdtemp.return_value = "/tmp/test_output"
        reader = OpenDataLoaderPDFReader()
        list(reader.lazy_load_data(file_path="doc.pdf"))

        mock_convert.assert_called_once()
        call_kwargs = mock_convert.call_args
        assert call_kwargs.kwargs["input_path"] == ["doc.pdf"]
        assert call_kwargs.kwargs["format"] == ["text"]
        assert call_kwargs.kwargs["quiet"] is False

    @patch("llama_index.readers.opendataloader_pdf.base.shutil.rmtree")
    @patch("llama_index.readers.opendataloader_pdf.base.Path.glob", return_value=[])
    @patch("llama_index.readers.opendataloader_pdf.base.tempfile.mkdtemp")
    @patch("opendataloader_pdf.convert")
    def test_none_values_omitted(
        self, mock_convert: MagicMock, mock_mkdtemp: MagicMock, *_: MagicMock
    ) -> None:
        """None values should not be passed to convert()."""
        mock_mkdtemp.return_value = "/tmp/test_output"
        reader = OpenDataLoaderPDFReader()
        list(reader.lazy_load_data(file_path="doc.pdf"))

        call_kwargs = mock_convert.call_args.kwargs
        assert "password" not in call_kwargs
        assert "table_method" not in call_kwargs
        assert "hybrid" not in call_kwargs

    @patch("llama_index.readers.opendataloader_pdf.base.shutil.rmtree")
    @patch("llama_index.readers.opendataloader_pdf.base.Path.glob", return_value=[])
    @patch("llama_index.readers.opendataloader_pdf.base.tempfile.mkdtemp")
    @patch("opendataloader_pdf.convert")
    def test_custom_kwargs_forwarded(
        self, mock_convert: MagicMock, mock_mkdtemp: MagicMock, *_: MagicMock
    ) -> None:
        mock_mkdtemp.return_value = "/tmp/test_output"
        reader = OpenDataLoaderPDFReader(
            format="markdown",
            sanitize=True,
            pages="1-5",
            use_struct_tree=True,
        )
        list(reader.lazy_load_data(file_path="doc.pdf"))

        call_kwargs = mock_convert.call_args.kwargs
        assert call_kwargs["format"] == ["markdown"]
        assert call_kwargs["sanitize"] is True
        assert call_kwargs["pages"] == "1-5"
        assert call_kwargs["use_struct_tree"] is True

    @patch("llama_index.readers.opendataloader_pdf.base.shutil.rmtree")
    @patch("llama_index.readers.opendataloader_pdf.base.Path.glob", return_value=[])
    @patch("llama_index.readers.opendataloader_pdf.base.tempfile.mkdtemp")
    @patch("opendataloader_pdf.convert")
    def test_page_separator_passed(
        self, mock_convert: MagicMock, mock_mkdtemp: MagicMock, *_: MagicMock
    ) -> None:
        """Page separator should be passed for all text-based formats."""
        mock_mkdtemp.return_value = "/tmp/test_output"
        reader = OpenDataLoaderPDFReader(split_pages=True)
        list(reader.lazy_load_data(file_path="doc.pdf"))

        call_kwargs = mock_convert.call_args.kwargs
        sep = "\n<<<ODL_PAGE_BREAK_%page-number%>>>\n"
        assert call_kwargs["markdown_page_separator"] == sep
        assert call_kwargs["text_page_separator"] == sep
        assert call_kwargs["html_page_separator"] == sep

    @patch("llama_index.readers.opendataloader_pdf.base.shutil.rmtree")
    @patch("llama_index.readers.opendataloader_pdf.base.Path.glob", return_value=[])
    @patch("llama_index.readers.opendataloader_pdf.base.tempfile.mkdtemp")
    @patch("opendataloader_pdf.convert")
    def test_no_separator_when_split_disabled(
        self, mock_convert: MagicMock, mock_mkdtemp: MagicMock, *_: MagicMock
    ) -> None:
        mock_mkdtemp.return_value = "/tmp/test_output"
        reader = OpenDataLoaderPDFReader(split_pages=False)
        list(reader.lazy_load_data(file_path="doc.pdf"))

        call_kwargs = mock_convert.call_args.kwargs
        assert call_kwargs["markdown_page_separator"] is None
        assert call_kwargs["text_page_separator"] is None
        assert call_kwargs["html_page_separator"] is None

    @patch("llama_index.readers.opendataloader_pdf.base.shutil.rmtree")
    @patch("llama_index.readers.opendataloader_pdf.base.Path.glob", return_value=[])
    @patch("llama_index.readers.opendataloader_pdf.base.tempfile.mkdtemp")
    @patch("opendataloader_pdf.convert")
    def test_multiple_files(
        self, mock_convert: MagicMock, mock_mkdtemp: MagicMock, *_: MagicMock
    ) -> None:
        mock_mkdtemp.return_value = "/tmp/test_output"
        reader = OpenDataLoaderPDFReader()
        list(reader.lazy_load_data(file_path=["a.pdf", "b.pdf"]))

        call_kwargs = mock_convert.call_args.kwargs
        assert call_kwargs["input_path"] == ["a.pdf", "b.pdf"]

    @patch("llama_index.readers.opendataloader_pdf.base.shutil.rmtree")
    @patch("llama_index.readers.opendataloader_pdf.base.Path.glob", return_value=[])
    @patch("llama_index.readers.opendataloader_pdf.base.tempfile.mkdtemp")
    @patch("opendataloader_pdf.convert")
    def test_path_object(
        self, mock_convert: MagicMock, mock_mkdtemp: MagicMock, *_: MagicMock
    ) -> None:
        mock_mkdtemp.return_value = "/tmp/test_output"
        reader = OpenDataLoaderPDFReader()
        list(reader.lazy_load_data(file_path=Path("doc.pdf")))

        call_kwargs = mock_convert.call_args.kwargs
        assert call_kwargs["input_path"] == ["doc.pdf"]


# ---------------------------------------------------------------------------
# TestSplitPages: Page splitting for text/markdown/html
# ---------------------------------------------------------------------------
class TestSplitPages:
    """Test page splitting logic for text-based formats."""

    def test_split_two_pages(self) -> None:
        reader = OpenDataLoaderPDFReader()
        content = (
            "\n<<<ODL_PAGE_BREAK_1>>>\nPage one content"
            "\n<<<ODL_PAGE_BREAK_2>>>\nPage two content"
        )
        docs = list(reader._split_into_pages(content, "test.pdf", "text"))
        assert len(docs) == 2
        assert docs[0].text == "Page one content"
        assert docs[0].metadata["page"] == 1
        assert docs[1].text == "Page two content"
        assert docs[1].metadata["page"] == 2

    def test_content_before_separator(self) -> None:
        reader = OpenDataLoaderPDFReader()
        content = "Before separator" "\n<<<ODL_PAGE_BREAK_2>>>\nPage two"
        docs = list(reader._split_into_pages(content, "test.pdf", "text"))
        assert len(docs) == 2
        assert docs[0].text == "Before separator"
        assert docs[0].metadata["page"] == 1
        assert docs[1].metadata["page"] == 2

    def test_empty_pages_skipped(self) -> None:
        reader = OpenDataLoaderPDFReader()
        content = (
            "\n<<<ODL_PAGE_BREAK_1>>>\nPage one"
            "\n<<<ODL_PAGE_BREAK_2>>>\n   "
            "\n<<<ODL_PAGE_BREAK_3>>>\nPage three"
        )
        docs = list(reader._split_into_pages(content, "test.pdf", "text"))
        assert len(docs) == 2
        assert docs[0].metadata["page"] == 1
        assert docs[1].metadata["page"] == 3

    def test_metadata_includes_source_and_format(self) -> None:
        reader = OpenDataLoaderPDFReader(format="markdown")
        content = "\n<<<ODL_PAGE_BREAK_1>>>\nContent"
        docs = list(reader._split_into_pages(content, "doc.pdf", "markdown"))
        assert docs[0].metadata["source"] == "doc.pdf"
        assert docs[0].metadata["format"] == "markdown"

    def test_extra_info_merged(self) -> None:
        reader = OpenDataLoaderPDFReader()
        content = "\n<<<ODL_PAGE_BREAK_1>>>\nContent"
        docs = list(
            reader._split_into_pages(content, "doc.pdf", "text", extra_info={"custom": "value"})
        )
        assert docs[0].metadata["custom"] == "value"


# ---------------------------------------------------------------------------
# TestSplitJsonPages: Page splitting for JSON format
# ---------------------------------------------------------------------------
class TestSplitJsonPages:
    """Test JSON page splitting logic."""

    def test_group_by_page_number(self) -> None:
        reader = OpenDataLoaderPDFReader(format="json")
        data = {
            "kids": [
                {"type": "paragraph", "page number": 1, "content": "p1"},
                {"type": "heading", "page number": 2, "content": "h2"},
                {"type": "paragraph", "page number": 1, "content": "p1b"},
            ]
        }
        docs = list(reader._split_json_into_pages(data, "test.pdf", "json"))
        assert len(docs) == 2
        assert docs[0].metadata["page"] == 1
        assert docs[1].metadata["page"] == 2

        page1_data = json.loads(docs[0].text)
        assert len(page1_data["kids"]) == 2

    def test_sorted_by_page_number(self) -> None:
        reader = OpenDataLoaderPDFReader(format="json")
        data = {
            "kids": [
                {"type": "paragraph", "page number": 3, "content": "p3"},
                {"type": "paragraph", "page number": 1, "content": "p1"},
            ]
        }
        docs = list(reader._split_json_into_pages(data, "test.pdf", "json"))
        assert docs[0].metadata["page"] == 1
        assert docs[1].metadata["page"] == 3

    def test_missing_page_number_defaults_to_1(self) -> None:
        """Elements without 'page number' should default to page 1."""
        reader = OpenDataLoaderPDFReader(format="json")
        data = {
            "kids": [
                {"type": "paragraph", "content": "no page field"},
            ]
        }
        docs = list(reader._split_json_into_pages(data, "test.pdf", "json"))
        assert len(docs) == 1
        assert docs[0].metadata["page"] == 1

    def test_extra_info_merged(self) -> None:
        reader = OpenDataLoaderPDFReader(format="json")
        data = {"kids": [{"type": "paragraph", "page number": 1, "content": "p"}]}
        docs = list(
            reader._split_json_into_pages(data, "test.pdf", "json", extra_info={"key": "val"})
        )
        assert docs[0].metadata["key"] == "val"


# ---------------------------------------------------------------------------
# TestMetadata: Verify metadata structure
# ---------------------------------------------------------------------------
class TestMetadata:
    """Test metadata content and hybrid key presence."""

    @patch("llama_index.readers.opendataloader_pdf.base.shutil.rmtree")
    @patch("llama_index.readers.opendataloader_pdf.base.tempfile.mkdtemp")
    @patch("llama_index.readers.opendataloader_pdf.base.Path.glob")
    def test_metadata_without_hybrid(
        self, mock_glob: MagicMock, mock_mkdtemp: MagicMock, *_: MagicMock
    ) -> None:
        mock_mkdtemp.return_value = "/tmp/test"
        mock_file = MagicMock(spec=Path)
        mock_file.with_suffix.return_value.name = "doc.pdf"
        mock_glob.return_value = [mock_file]

        with (
            patch("opendataloader_pdf.convert"),
            patch("builtins.open", mock_open(read_data="content")),
        ):
            reader = OpenDataLoaderPDFReader(split_pages=False)
            docs = list(reader.lazy_load_data(file_path="doc.pdf"))

            assert "hybrid" not in docs[0].metadata
            assert docs[0].metadata["source"] == "doc.pdf"
            assert docs[0].metadata["format"] == "text"

    @patch("llama_index.readers.opendataloader_pdf.base.shutil.rmtree")
    @patch("llama_index.readers.opendataloader_pdf.base.tempfile.mkdtemp")
    @patch("llama_index.readers.opendataloader_pdf.base.Path.glob")
    def test_metadata_with_hybrid(
        self, mock_glob: MagicMock, mock_mkdtemp: MagicMock, *_: MagicMock
    ) -> None:
        mock_mkdtemp.return_value = "/tmp/test"
        mock_file = MagicMock(spec=Path)
        mock_file.with_suffix.return_value.name = "doc.pdf"
        mock_glob.return_value = [mock_file]

        with (
            patch("opendataloader_pdf.convert"),
            patch("builtins.open", mock_open(read_data="content")),
        ):
            reader = OpenDataLoaderPDFReader(split_pages=False, hybrid="docling-fast")
            docs = list(reader.lazy_load_data(file_path="doc.pdf"))

            assert docs[0].metadata["hybrid"] == "docling-fast"


# ---------------------------------------------------------------------------
# TestExtraInfo: extra_info merges into metadata
# ---------------------------------------------------------------------------
class TestExtraInfo:
    """Test that extra_info dict is merged into document metadata."""

    @patch("llama_index.readers.opendataloader_pdf.base.shutil.rmtree")
    @patch("llama_index.readers.opendataloader_pdf.base.tempfile.mkdtemp")
    @patch("llama_index.readers.opendataloader_pdf.base.Path.glob")
    def test_extra_info_merged(
        self, mock_glob: MagicMock, mock_mkdtemp: MagicMock, *_: MagicMock
    ) -> None:
        mock_mkdtemp.return_value = "/tmp/test"
        mock_file = MagicMock(spec=Path)
        mock_file.with_suffix.return_value.name = "doc.pdf"
        mock_glob.return_value = [mock_file]

        with (
            patch("opendataloader_pdf.convert"),
            patch("builtins.open", mock_open(read_data="content")),
        ):
            reader = OpenDataLoaderPDFReader(split_pages=False)
            docs = list(
                reader.lazy_load_data(
                    file_path="doc.pdf",
                    extra_info={"custom_key": "custom_value"},
                )
            )

            assert docs[0].metadata["custom_key"] == "custom_value"
            assert docs[0].metadata["source"] == "doc.pdf"


# ---------------------------------------------------------------------------
# TestTempCleanup: Temp directory cleanup behavior
# ---------------------------------------------------------------------------
class TestTempCleanup:
    """Test temp directory cleanup."""

    @patch("llama_index.readers.opendataloader_pdf.base.shutil.rmtree")
    @patch("llama_index.readers.opendataloader_pdf.base.Path.glob", return_value=[])
    @patch("llama_index.readers.opendataloader_pdf.base.tempfile.mkdtemp")
    @patch("opendataloader_pdf.convert")
    def test_temp_dir_cleaned(
        self,
        mock_convert: MagicMock,
        mock_mkdtemp: MagicMock,
        mock_glob: MagicMock,
        mock_rmtree: MagicMock,
    ) -> None:
        mock_mkdtemp.return_value = "/tmp/test_output"
        reader = OpenDataLoaderPDFReader()
        list(reader.lazy_load_data(file_path="doc.pdf"))

        mock_rmtree.assert_called_once_with("/tmp/test_output", ignore_errors=True)

    @patch("llama_index.readers.opendataloader_pdf.base.shutil.rmtree")
    @patch("llama_index.readers.opendataloader_pdf.base.Path.glob", return_value=[])
    @patch("llama_index.readers.opendataloader_pdf.base.tempfile.mkdtemp")
    @patch("opendataloader_pdf.convert")
    def test_temp_dir_kept_for_external_images(
        self,
        mock_convert: MagicMock,
        mock_mkdtemp: MagicMock,
        mock_glob: MagicMock,
        mock_rmtree: MagicMock,
    ) -> None:
        """Temp dir should NOT be cleaned when images are stored there."""
        mock_mkdtemp.return_value = "/tmp/test_output"
        reader = OpenDataLoaderPDFReader(image_output="external")
        list(reader.lazy_load_data(file_path="doc.pdf"))

        mock_rmtree.assert_not_called()


# ---------------------------------------------------------------------------
# TestHybridErrors: Error handling behavior
# ---------------------------------------------------------------------------
class TestHybridErrors:
    """Test error handling differs by hybrid mode."""

    @patch("llama_index.readers.opendataloader_pdf.base.shutil.rmtree")
    @patch("llama_index.readers.opendataloader_pdf.base.tempfile.mkdtemp")
    @patch("opendataloader_pdf.convert", side_effect=RuntimeError("fail"))
    def test_hybrid_error_reraises(
        self, mock_convert: MagicMock, mock_mkdtemp: MagicMock, *_: MagicMock
    ) -> None:
        mock_mkdtemp.return_value = "/tmp/test"
        reader = OpenDataLoaderPDFReader(hybrid="docling-fast")
        with pytest.raises(RuntimeError, match="fail"):
            list(reader.lazy_load_data(file_path="doc.pdf"))

    @patch("llama_index.readers.opendataloader_pdf.base.shutil.rmtree")
    @patch("llama_index.readers.opendataloader_pdf.base.tempfile.mkdtemp")
    @patch("opendataloader_pdf.convert", side_effect=RuntimeError("fail"))
    def test_non_hybrid_error_silent(
        self, mock_convert: MagicMock, mock_mkdtemp: MagicMock, *_: MagicMock
    ) -> None:
        mock_mkdtemp.return_value = "/tmp/test"
        reader = OpenDataLoaderPDFReader()
        docs = list(reader.lazy_load_data(file_path="doc.pdf"))
        assert docs == []


# ---------------------------------------------------------------------------
# TestEmptyFile: Empty output handling
# ---------------------------------------------------------------------------
class TestEmptyFile:
    """Test behavior when output file is empty."""

    @patch("llama_index.readers.opendataloader_pdf.base.shutil.rmtree")
    @patch("llama_index.readers.opendataloader_pdf.base.tempfile.mkdtemp")
    @patch("llama_index.readers.opendataloader_pdf.base.Path.glob")
    def test_empty_file_returns_empty(
        self, mock_glob: MagicMock, mock_mkdtemp: MagicMock, *_: MagicMock
    ) -> None:
        mock_mkdtemp.return_value = "/tmp/test"
        mock_file = MagicMock(spec=Path)
        mock_file.with_suffix.return_value.name = "doc.pdf"
        mock_glob.return_value = [mock_file]

        with (
            patch("opendataloader_pdf.convert"),
            patch("builtins.open", mock_open(read_data="")),
        ):
            reader = OpenDataLoaderPDFReader(split_pages=True)
            docs = list(reader.lazy_load_data(file_path="doc.pdf"))
            assert docs == []


# ---------------------------------------------------------------------------
# TestSinglePageNoSeparator: Content without page separators
# ---------------------------------------------------------------------------
class TestSinglePageNoSeparator:
    """Test that content without separators is treated as page 1."""

    def test_no_separator_becomes_page_1(self) -> None:
        reader = OpenDataLoaderPDFReader()
        content = "Just plain content without any separator"
        docs = list(reader._split_into_pages(content, "test.pdf", "text"))
        assert len(docs) == 1
        assert docs[0].text == "Just plain content without any separator"
        assert docs[0].metadata["page"] == 1


# ---------------------------------------------------------------------------
# TestImportError: Missing opendataloader_pdf dependency
# ---------------------------------------------------------------------------
class TestImportError:
    """Test behavior when opendataloader_pdf is not installed."""

    @patch("llama_index.readers.opendataloader_pdf.base.tempfile.mkdtemp")
    @patch("llama_index.readers.opendataloader_pdf.base.shutil.rmtree")
    def test_import_error_propagates(self, mock_rmtree: MagicMock, mock_mkdtemp: MagicMock) -> None:
        mock_mkdtemp.return_value = "/tmp/test"
        reader = OpenDataLoaderPDFReader()

        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "opendataloader_pdf":
                raise ImportError("No module named 'opendataloader_pdf'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="opendataloader_pdf"):
                list(reader.lazy_load_data(file_path="doc.pdf"))


# ---------------------------------------------------------------------------
# TestFileNotFound: Missing input file raises FileNotFoundError
# ---------------------------------------------------------------------------
class TestFileNotFound:
    """Test that missing input files raise FileNotFoundError."""

    @pytest.fixture(autouse=True)
    def _real_path_exists(self, monkeypatch):
        """Restore real Path.exists so file-not-found is testable."""
        monkeypatch.setattr("pathlib.Path.exists", _original_path_exists)

    def test_single_missing_file(self) -> None:
        reader = OpenDataLoaderPDFReader()
        with pytest.raises(
            FileNotFoundError,
            match=r"Input path does not exist: nonexistent\.pdf",
        ):
            list(reader.lazy_load_data(file_path="nonexistent.pdf"))

    def test_missing_file_in_list(self) -> None:
        reader = OpenDataLoaderPDFReader()
        with pytest.raises(
            FileNotFoundError,
            match=r"Input path does not exist: nonexistent\.pdf",
        ):
            list(reader.lazy_load_data(file_path=["nonexistent.pdf"]))


# ---------------------------------------------------------------------------
# TestJavaCheck: Missing Java raises RuntimeError
# ---------------------------------------------------------------------------
class TestJavaCheck:
    """Test that missing Java runtime raises RuntimeError."""

    @patch(
        "llama_index.readers.opendataloader_pdf.base._java_available",
        return_value=False,
    )
    def test_no_java_raises_runtime_error(self, _: MagicMock) -> None:
        reader = OpenDataLoaderPDFReader()
        with pytest.raises(RuntimeError, match="Java is not found"):
            list(reader.lazy_load_data(file_path="doc.pdf"))
