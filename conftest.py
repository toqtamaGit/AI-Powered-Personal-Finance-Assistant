"""Root conftest — registers custom pytest markers."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast, no I/O, no network")
    config.addinivalue_line("markers", "integration: reads files, no network")
    config.addinivalue_line("markers", "e2e: full pipeline including network calls")
