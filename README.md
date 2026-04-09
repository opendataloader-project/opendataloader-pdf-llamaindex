<!-- AI-AGENT-SUMMARY
name: opendataloader-pdf-llamaindex
category: LlamaIndex reader, PDF extraction for RAG
license: Apache-2.0
solves: [Load PDFs as LlamaIndex Document objects for RAG pipelines, structured PDF extraction with correct reading order and table preservation]
input: PDF files (digital, tagged)
output: LlamaIndex Document objects (text, Markdown, JSON with bounding boxes, HTML)
sdk: Python
requirements: Python 3.10+, Java 11+
key-differentiators: [LlamaIndex-native BasePydanticReader, per-page Document splitting, SimpleDirectoryReader file_extractor support, all opendataloader-pdf extraction features]
-->

# opendataloader-pdf-llamaindex

LlamaIndex reader for [OpenDataLoader PDF](https://github.com/opendataloader-project/opendataloader-pdf) — parse PDFs into structured `Document` objects for RAG pipelines.

For the full feature set of the core engine (hybrid AI mode, OCR, formula extraction, benchmarks, accessibility), see the [OpenDataLoader PDF documentation](https://opendataloader.org/docs).

[![PyPI version](https://img.shields.io/pypi/v/opendataloader-pdf-llamaindex.svg)](https://pypi.org/project/opendataloader-pdf-llamaindex/)
[![License](https://img.shields.io/pypi/l/opendataloader-pdf-llamaindex.svg)](https://github.com/opendataloader-project/opendataloader-pdf-llamaindex/blob/main/LICENSE)

## Features

- **Accurate reading order** — XY-Cut++ algorithm handles multi-column layouts correctly
- **Table extraction** — Preserves table structure in output
- **Multiple formats** — Text, Markdown, JSON (with bounding boxes), HTML
- **Per-page splitting** — Each page becomes a separate `Document` with page number metadata
- **AI safety** — Built-in prompt injection filtering (hidden text, off-page content, invisible layers)
- **100% local** — No cloud APIs, your documents never leave your machine
- **Fast** — Rule-based extraction, no GPU required

## Requirements

- Python >= 3.10
- Java 11+ available on system `PATH`

Verify Java is installed:

```bash
java -version
```

## Installation

```bash
pip install -U opendataloader-pdf-llamaindex
```

## Quick Start

```python
from llama_index.readers.opendataloader_pdf import OpenDataLoaderPDFReader

reader = OpenDataLoaderPDFReader(format="text")
documents = reader.load_data(file_path="document.pdf")

print(documents[0].text)
print(documents[0].metadata)
# {'source': 'document.pdf', 'format': 'text', 'page': 1}
```

## SimpleDirectoryReader Integration

Use with LlamaIndex's `SimpleDirectoryReader` via the `file_extractor` parameter:

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.opendataloader_pdf import OpenDataLoaderPDFReader

reader = SimpleDirectoryReader(
    input_dir="./documents",
    file_extractor={".pdf": OpenDataLoaderPDFReader(format="markdown")}
)
documents = reader.load_data()
```

## Usage Examples

### Output Formats

```python
from llama_index.readers.opendataloader_pdf import OpenDataLoaderPDFReader

# Plain text (default) — best for simple RAG
reader = OpenDataLoaderPDFReader(format="text")

# Markdown — preserves headings, lists, tables
reader = OpenDataLoaderPDFReader(format="markdown")

# JSON — structured data with bounding boxes for source citations
reader = OpenDataLoaderPDFReader(format="json")

# HTML — styled output
reader = OpenDataLoaderPDFReader(format="html")
```

### Tagged PDF Support

For accessible PDFs with structure tags (common in government/legal documents):

```python
reader = OpenDataLoaderPDFReader(use_struct_tree=True)
```

### Table Detection

```python
reader = OpenDataLoaderPDFReader(
    format="markdown",
    table_method="cluster"  # Better for borderless tables
)
```

### Sensitive Data Sanitization

```python
reader = OpenDataLoaderPDFReader(sanitize=True)
# Replaces emails, phone numbers, IPs, credit cards, URLs with placeholders
```

### Page Selection

```python
reader = OpenDataLoaderPDFReader(pages="1,3,5-7")
```

### Headers and Footers

```python
reader = OpenDataLoaderPDFReader(include_header_footer=True)
```

### Password-Protected PDFs

```python
reader = OpenDataLoaderPDFReader(password="secret")
```

### Image Handling

```python
# Embed images as Base64 in output
reader = OpenDataLoaderPDFReader(image_output="embedded")

# Save images to external files
reader = OpenDataLoaderPDFReader(
    image_output="external",
    image_dir="./extracted_images"
)
```

### Hybrid AI Mode

For higher accuracy on complex documents (requires a running hybrid backend):

```python
reader = OpenDataLoaderPDFReader(
    hybrid="docling-fast",
    hybrid_fallback=True  # Fall back to Java on backend failure
)
```

## RAG Pipeline Example

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.opendataloader_pdf import OpenDataLoaderPDFReader

# Load PDFs
reader = SimpleDirectoryReader(
    input_dir="./documents",
    file_extractor={".pdf": OpenDataLoaderPDFReader(format="markdown")}
)
documents = reader.load_data()

# Build index and query
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What are the key findings?")
print(response)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | `str` | `"text"` | Output format: `"text"`, `"markdown"`, `"json"`, `"html"` |
| `split_pages` | `bool` | `True` | Split output into separate Documents per page |
| `quiet` | `bool` | `False` | Suppress CLI logging output |
| `content_safety_off` | `list[str]` | `None` | Safety filters to disable: `"all"`, `"hidden-text"`, `"off-page"`, `"tiny"`, `"hidden-ocg"` |
| `password` | `str` | `None` | Password for encrypted PDFs |
| `keep_line_breaks` | `bool` | `False` | Preserve original line breaks |
| `replace_invalid_chars` | `str` | `None` | Replacement for unrecognized characters |
| `use_struct_tree` | `bool` | `False` | Use PDF structure tree (tagged PDFs) |
| `table_method` | `str` | `None` | `"default"` (border-based) or `"cluster"` (border + cluster) |
| `reading_order` | `str` | `None` | `"off"` or `"xycut"` (default when not specified) |
| `image_output` | `str` | `"off"` | `"off"`, `"embedded"` (Base64), `"external"` (files) |
| `image_format` | `str` | `None` | `"png"` or `"jpeg"` |
| `image_dir` | `str` | `None` | Directory for external images |
| `sanitize` | `bool` | `False` | Mask emails, phones, IPs, credit cards, URLs |
| `pages` | `str` | `None` | Pages to extract, e.g., `"1,3,5-7"` |
| `include_header_footer` | `bool` | `False` | Include page headers and footers |
| `detect_strikethrough` | `bool` | `False` | Detect strikethrough text (experimental) |
| `hybrid` | `str` | `None` | Hybrid AI backend: `"docling-fast"` |
| `hybrid_mode` | `str` | `None` | `"auto"` (complex pages only) or `"full"` (all pages) |
| `hybrid_url` | `str` | `None` | Custom backend server URL |
| `hybrid_timeout` | `str` | `None` | Backend timeout in milliseconds |
| `hybrid_fallback` | `bool` | `False` | Fall back to Java on backend failure |

## Document Metadata

Each `Document` includes metadata:

**With `split_pages=True` (default):**

```python
{"source": "document.pdf", "format": "text", "page": 1}
```

**With `split_pages=False`:**

```python
{"source": "document.pdf", "format": "text"}
```

**With hybrid mode:**

```python
{"source": "document.pdf", "format": "text", "page": 1, "hybrid": "docling-fast"}
```
