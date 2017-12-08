import requests
from time import sleep
from selenium.webdriver.support.ui import Select


def test_post_analysis_job_request(scanomatic, browser):

    browser_name = browser.capabilities['browserName'].replace(' ', '_')
    payload = requests.get(
        scanomatic + '/api/analysis/instructions/testproject/test_ccc'
    ).json()
    if payload.get('instructions'):
        assert False, (
            "Test environment is not clean. There's stuff in the output folder"
            .format(payload)
        )

    browser.get(scanomatic + '/analysis')

    elem = browser.find_element_by_id('compilation')
    elem.send_keys('')
    elem.send_keys('testproject/testproject.project.compilation')

    elem = Select(browser.find_element_by_id('ccc-selection'))
    elem.select_by_visible_text('Testum testare, testis')

    elem = browser.find_element_by_id('analysis-directory')
    elem.send_keys('test_ccc_{}'.format(browser_name))

    elem = browser.find_element_by_css_selector(
        'label[for=chain-analysis-request]'
    )
    elem.click()

    browser.find_element_by_id('submit-button').click()

    uri = (
        scanomatic +
        '/api/analysis/instructions/testproject/test_ccc_{}'.format(
            browser_name)
    )
    tries = 0
    while tries < 100:
        payload = requests.get(uri).json()
        if payload.get('instructions'):
            assert payload['instructions'].get('ccc') == 'TEST'
            return
        else:
            tries += 1
            sleep(0.5)
    assert False, "Time out waiting for {} to show analysis started with expected ccc".format(
        uri
    )
