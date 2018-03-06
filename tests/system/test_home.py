from __future__ import absolute_import


def test_title(scanomatic, browser):
    browser.get(scanomatic)
    assert "Scan-o-Matic" in browser.title
