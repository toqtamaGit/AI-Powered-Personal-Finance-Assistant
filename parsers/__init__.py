"""Bank statement parsers and normalizers."""

from parsers.base import BaseParser, BaseNormalizer
from parsers.kaspi import KaspiParser
from parsers.freedom import FreedomParser
from parsers.normalize_kaspi import KaspiNormalizer
from parsers.normalize_freedom import FreedomNormalizer

# Registry: bank_name → (parser_class, normalizer_class)
REGISTRY = {
    "Kaspi Bank": (KaspiParser, KaspiNormalizer),
    "Freedom Bank Kazakhstan": (FreedomParser, FreedomNormalizer),
}

__all__ = [
    "BaseParser",
    "BaseNormalizer",
    "KaspiParser",
    "FreedomParser",
    "KaspiNormalizer",
    "FreedomNormalizer",
    "REGISTRY",
]
