"""Backward-compatible re-export — implementation moved to parsers.normalize_kaspi."""

from parsers.normalize_kaspi import (  # noqa: F401
    KaspiNormalizer,
    parse_details,
    normalize_csv,
    main,
    _PURCHASE_OPS,
    BUSINESS_TYPE_RE,
    KNOWN_PLACES,
    GENERIC_PLACES as _GENERIC_PLACES,
    TYPE_MAP as _TYPE_MAP,
)

if __name__ == '__main__':
    main()
