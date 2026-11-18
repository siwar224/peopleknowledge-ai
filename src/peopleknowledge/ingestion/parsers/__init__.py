"""Parsing stage: turn raw source files (PDF, DOCX, CSV) into structured documents.

Importing the parser modules registers them (``@register_parser``), so that
``get_parser(".pdf")`` works as soon as this package is imported.
"""

from . import csv_parser, docx_parser, pdf_parser  # noqa: F401  (registration side effect)
from .base import ParsingError, UnsupportedFormatError, get_parser, supported_extensions

__all__ = ["ParsingError", "UnsupportedFormatError", "get_parser", "supported_extensions"]
