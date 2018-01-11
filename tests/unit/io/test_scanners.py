from __future__ import absolute_import
import pytest
from scanomatic.io.scanners import Scanners


@pytest.fixture
def scanners():
    return Scanners()


def test_has_test_scanner(scanners):
    assert scanners.has_scanner("Test")


def test_not_having_unkown_scanner(scanners):
    assert scanners.has_scanner("Unknown") is False


def test_getting_scanner(scanners):
    assert scanners.get("Test") == {
        "name": "Test",
        "power": False,
        "owner": None,
    }


def test_get_free(scanners):
    assert scanners.get_free() == [
        {
            "name": "Test",
            "power": False,
            "owner": None,
        },
    ]


def test_get_all(scanners):
    assert scanners.get_free() == [
        {
            "name": "Test",
            "power": False,
            "owner": None,
        },
    ]
