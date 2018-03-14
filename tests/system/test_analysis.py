from __future__ import absolute_import

from time import sleep
from warnings import warn

import pytest
import requests
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select


@pytest.yield_fixture(autouse=True)
def cleanup_rpc(scanomatic):
    # Nothing before
    yield
    # Remove all jobs after
    jobs = requests.get(scanomatic + '/api/status/jobs').json()['jobs']
    for job in jobs:
        job_id = job['id']
        response = requests.get(scanomatic + '/api/job/{}/stop'.format(job_id))
        if response.status_code != 200:
            warn('Could not terminate job {}'.format(job_id))


def test_post_analysis_job_request(scanomatic, browser):

    # FILLING IN THE FORM

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

    # To better ensure the '/root/' is in place in the input
    elem.send_keys('', Keys.BACKSPACE)
    sleep(0.1)

    elem.send_keys('testproject/testproject.project.compilation')

    elem = Select(browser.find_element_by_id('ccc-selection'))
    elem.select_by_visible_text('S. cerevisiae, Zackrisson et. al. 2016')

    elem = browser.find_element_by_id('analysis-directory')
    elem.send_keys('test_ccc_{}'.format(browser_name))

    elem = browser.find_element_by_css_selector(
        'label[for=chain-analysis-request]'
    )
    elem.click()

    browser.find_element_by_id('submit-button').click()
