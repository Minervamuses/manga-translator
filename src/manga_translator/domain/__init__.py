"""Versioned, persistence-safe domain models."""

from .models import PageDocument
from .serialization import canonical_document_bytes, parse_document

__all__ = ["PageDocument", "canonical_document_bytes", "parse_document"]
