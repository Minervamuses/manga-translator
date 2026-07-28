"""Non-destructive text variants and Unicode segmentation."""

from .normalize import normalize_text
from .variants import TextVariants, grapheme_clusters

__all__ = ["TextVariants", "grapheme_clusters", "normalize_text"]
