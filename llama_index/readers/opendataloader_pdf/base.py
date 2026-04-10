"""OpenDataLoader PDF reader for LlamaIndex."""

import functools
import json
import logging
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Union

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _java_available() -> bool:
    """Return True if a working Java runtime is found on the system PATH."""
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


_FORMAT_TO_EXT = {
    "json": "json",
    "text": "txt",
    "html": "html",
    "markdown": "md",
}


class OpenDataLoaderPDFReader(BasePydanticReader):
    """OpenDataLoader PDF Reader.

    Uses `opendataloader-pdf <https://github.com/opendataloader-project/opendataloader-pdf>`_
    to extract content from PDFs into Markdown, JSON, HTML, or plain text
    for usage in LlamaIndex pipelines.

    OpenDataLoader PDF runs 100% locally — no cloud APIs or GPU required.

    Args:
        format: Output format. One of ``"text"``, ``"json"``, ``"html"``,
            ``"markdown"``. Defaults to ``"text"``.
        split_pages: If True, split output into separate Documents per page.
            Defaults to True.
        quiet: Suppress CLI logging output. Defaults to False.
        hybrid: Backend for hybrid AI extraction. None = Java-only (default).
            Values: ``"docling-fast"``.
    """

    # --- BEGIN SYNCED PARAMS ---
    format: str = "text"
    quiet: bool = False
    content_safety_off: Optional[List[str]] = None
    password: Optional[str] = None
    keep_line_breaks: bool = False
    replace_invalid_chars: Optional[str] = None
    use_struct_tree: bool = False
    table_method: Optional[str] = None
    reading_order: Optional[str] = None
    image_output: str = "off"
    image_format: Optional[str] = None
    image_dir: Optional[str] = None
    sanitize: bool = False
    pages: Optional[str] = None
    include_header_footer: bool = False
    detect_strikethrough: bool = False
    hybrid: Optional[str] = None
    hybrid_mode: Optional[str] = None
    hybrid_url: Optional[str] = None
    hybrid_timeout: Optional[str] = None
    hybrid_fallback: bool = False
    # --- END SYNCED PARAMS ---
    split_pages: bool = True

    # Internal separator used for page splitting.
    _PAGE_SPLIT_SEPARATOR: ClassVar[str] = "\n<<<ODL_PAGE_BREAK_%page-number%>>>\n"

    def _get_page_separator(self) -> Optional[str]:
        """Return the page separator string when split_pages is enabled."""
        if self.split_pages:
            return self._PAGE_SPLIT_SEPARATOR
        return None

    def _split_into_pages(
        self,
        content: str,
        source_name: str,
        fmt: str,
        extra_info: Optional[Dict] = None,
    ) -> Iterable[Document]:
        """Split text/markdown/html content by page separator."""
        separator_pattern = re.escape(self._PAGE_SPLIT_SEPARATOR).replace(
            re.escape("%page-number%"), r"(\d+)"
        )

        parts = re.split(separator_pattern, content)

        # Handle content before first separator (treat as page 1).
        if parts[0].strip():
            yield Document(
                text=parts[0].strip(),
                metadata={
                    **(extra_info or {}),
                    "source": source_name,
                    "format": fmt,
                    "page": 1,
                    **({"hybrid": self.hybrid} if self.hybrid else {}),
                },
            )

        # Process remaining parts: (page_num, content) pairs.
        for i in range(1, len(parts), 2):
            page_num = int(parts[i])
            if i + 1 < len(parts):
                page_content = parts[i + 1].strip()
                if page_content:
                    yield Document(
                        text=page_content,
                        metadata={
                            **(extra_info or {}),
                            "source": source_name,
                            "format": fmt,
                            "page": page_num,
                            **({"hybrid": self.hybrid} if self.hybrid else {}),
                        },
                    )

    def _split_json_into_pages(
        self,
        data: Dict[str, Any],
        source_name: str,
        fmt: str,
        extra_info: Optional[Dict] = None,
    ) -> Iterable[Document]:
        """Split JSON content by page number field."""
        pages: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        for element in data.get("kids", []):
            page_num = element.get("page number", 1)
            pages[page_num].append(element)

        for page_num in sorted(pages.keys()):
            page_data = {
                "page number": page_num,
                "kids": pages[page_num],
            }
            page_content = json.dumps(page_data, ensure_ascii=False)

            yield Document(
                text=page_content,
                metadata={
                    **(extra_info or {}),
                    "source": source_name,
                    "format": fmt,
                    "page": page_num,
                    **({"hybrid": self.hybrid} if self.hybrid else {}),
                },
            )

    def lazy_load_data(
        self,
        file_path: Union[str, Path, Iterable[Union[str, Path]]],
        extra_info: Optional[Dict] = None,
    ) -> Iterable[Document]:
        """Lazily load documents from PDF file(s).

        Args:
            file_path: One or more PDF file paths, or a directory.
            extra_info: Additional metadata merged into each document.

        Yields:
            LlamaIndex Document for each converted output (or per page
            when ``split_pages`` is enabled).
        """
        from opendataloader_pdf import convert

        fmt = self.format.lower()
        if fmt not in _FORMAT_TO_EXT:
            raise ValueError(
                f"Invalid format '{self.format}'. "
                f"Valid options: {list(_FORMAT_TO_EXT.keys())}"
            )

        if isinstance(file_path, (str, Path)):
            paths = [str(file_path)]
        else:
            paths = [str(p) for p in file_path]

        for p in paths:
            if not Path(p).exists():
                raise FileNotFoundError(f"Input path does not exist: {p}")

        if not _java_available():
            raise RuntimeError(
                "Java is not found on the system PATH. "
                "OpenDataLoader PDF requires Java 11+. "
                "Install Java from https://adoptium.net/ and ensure "
                "'java' is on your system PATH. "
                "Verify with: java -version"
            )

        ext = _FORMAT_TO_EXT[fmt]

        try:
            output_dir = tempfile.mkdtemp()
        except OSError as e:
            logger.exception("Failed to create temp directory")
            return

        try:
            try:
                page_sep = self._get_page_separator()

                # --- BEGIN SYNCED CONVERT KWARGS ---
                convert_kwargs: Dict[str, Any] = {
                    "format": [fmt],
                    "quiet": self.quiet,
                    "content_safety_off": self.content_safety_off,
                    "password": self.password,
                    "keep_line_breaks": self.keep_line_breaks,
                    "replace_invalid_chars": self.replace_invalid_chars,
                    "use_struct_tree": self.use_struct_tree,
                    "table_method": self.table_method,
                    "reading_order": self.reading_order,
                    "image_output": self.image_output,
                    "image_format": self.image_format,
                    "image_dir": self.image_dir,
                    "sanitize": self.sanitize,
                    "pages": self.pages,
                    "include_header_footer": self.include_header_footer,
                    "detect_strikethrough": self.detect_strikethrough,
                    "hybrid": self.hybrid,
                    "hybrid_mode": self.hybrid_mode,
                    "hybrid_url": self.hybrid_url,
                    "hybrid_timeout": self.hybrid_timeout,
                    "hybrid_fallback": self.hybrid_fallback,
                }
                # --- END SYNCED CONVERT KWARGS ---
                # Omit None values so the core engine applies its own defaults.
                convert_kwargs = {
                    k: v for k, v in convert_kwargs.items() if v is not None
                }

                convert(
                    input_path=paths,
                    output_dir=output_dir,
                    **convert_kwargs,
                    markdown_page_separator=page_sep,
                    text_page_separator=page_sep,
                    html_page_separator=page_sep,
                )
            except Exception as e:
                if self.hybrid:
                    raise
                logger.exception("Error during conversion")
                return

            output_path = Path(output_dir)
            for file in sorted(output_path.glob(f"*.{ext}")):
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()

                source_name = file.with_suffix(".pdf").name

                if self.split_pages:
                    if fmt == "json":
                        data = json.loads(content)
                        yield from self._split_json_into_pages(
                            data, source_name, fmt, extra_info
                        )
                    else:
                        yield from self._split_into_pages(
                            content, source_name, fmt, extra_info
                        )
                else:
                    yield Document(
                        text=content,
                        metadata={
                            **(extra_info or {}),
                            "source": source_name,
                            "format": fmt,
                            **({"hybrid": self.hybrid} if self.hybrid else {}),
                        },
                    )
        except Exception:
            logger.debug("Error processing output files", exc_info=True)
            raise
        finally:
            if self.image_output == "external" and self.image_dir is None:
                logger.info(
                    "Extracted images retained in temp directory: %s",
                    output_dir,
                )
            else:
                shutil.rmtree(output_dir, ignore_errors=True)
