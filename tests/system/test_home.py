def test_title(scanomatic, browser):
    browser.get(scanomatic)
    assert "Scan-o-Matic" in browser.title
