"""Backward-compatible re-export — implementation moved to parsers.normalize_freedom."""

from parsers.normalize_freedom import (  # noqa: F401
    FreedomNormalizer,
    parse_freedom_details,
    normalize_hyphens,
    normalize_csv,
    extract_city_from_text,
    main,
    _BUSINESS_OPS,
    BUSINESS_TYPE_RE,
    KNOWN_PLACES,
)

if __name__ == '__main__':
    main()
