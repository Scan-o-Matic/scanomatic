from __future__ import absolute_import

import json

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skip-slow",
        action="store_true",
        default=False,
        help="skip tests marked as slow"
    )
    parser.addoption(
        '--browser',
        action='store',
        choices=['firefox', 'chrome'],
        default='firefox',
    )
    parser.addoption(
        '--scanomatic-url',
        action='store',
        help='''Run system tests againt the provided URL instead of starting
                scan-o-matic with docker-compose''',
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(
            reason="skipped due to '--skip-slow' flag")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def _load_measurements(path):
    with path.open('r') as f:
        return json.load(f)


@pytest.fixture
def good_ccc_measurements(pytestconfig):
    return _load_measurements(
        pytestconfig.rootdir.join('tests/fixtures/good_ccc_measurements.json')
    )


@pytest.fixture
def badslope_ccc_measurements(pytestconfig):
    return _load_measurements(
        pytestconfig.rootdir.join('tests/fixtures/badslope_ccc_measurements.json')
    )


@pytest.fixture
def full_ccc_measurements(pytestconfig):
    return _load_measurements(
        pytestconfig.rootdir.join('tests/fixtures/full_ccc_measurements.json')
    )
