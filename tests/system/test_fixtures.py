def test_new_fixture_name(scanomatic, browser):
    """ Regression test for issue #173 """
    browser.get(scanomatic + '/fixtures')
    browser.find_element_by_id('add-fixture').click()
    element = browser.find_element_by_id('new-fixture-name')
    element.send_keys('Exterminate')
    assert element.get_attribute('value') == "Exterminate"
