from __future__ import absolute_import
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skip-slow",
        action="store_true",
        default=False,
        help="skip tests marked as slow"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(
            reason="skipped due to '--skip-slow' flag")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
