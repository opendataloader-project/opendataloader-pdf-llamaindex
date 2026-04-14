# Changelog

## [Unreleased]

## [0.0.3] - 2026-04-14

### Added
- OpenDataLoaderPDFReaderDemo.ipynb demo notebook (Colab-ready, 44 cells)
- GitHub issue templates (bug report, feature request, question) and PR template
- SECURITY.md, CODEOWNERS, .editorconfig
- Lint CI workflow (ruff + black, enforced on every PR)

### Changed
- Reformatted existing code to pass ruff/black (line-length=100)

### Fixed
- RAG pipeline cell in demo notebook no longer requires HF_TOKEN

## [0.0.2] - 2026-04-10

### Fixed
- Clear error messages for missing input files and unavailable Java runtime

## [0.0.1] - 2026-04-10

### Added
- `OpenDataLoaderPDFReader` with 21 extraction parameters and per-page splitting
- `SimpleDirectoryReader` `file_extractor` support
- Hybrid AI extraction mode support (docling-fast backend)
- Unit tests with mock-based testing and pytest-socket network isolation
- Integration tests with real Java engine and PDF files
- CI/CD workflows: test (PR), test-full (multi-platform), release (PyPI)
