from __future__ import absolute_import

import pytest
from warnings import warn
import requests
from time import sleep
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys


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


def assert_has_job(scanomatic, job_settings):
    uri = scanomatic + '/api/status/queue'
    payload = requests.get(uri).json()
    if payload.get('queue', None):
        for item in payload.get('queue'):
            if item.get('type') == 'Analysis':
                model = item.get('content_model')
                has_compilation = (
                    job_settings['compilation'] in model['compilation']
                )
                has_ccc = (
                    model['cell_count_calibration_id'] ==
                    job_settings['cell_count_calibration_id'])
                has_output = (
                    model['output_directory'] ==
                    job_settings['output_directory']
                )
                if has_output:
                    assert has_compilation, (
                        "Job used unexpected compilation '{}'".format(
                            model.get('compilation')
                        )
                    )
                    assert has_ccc, (
                        "Job used unexpected ccc '{}'".format(
                            model.get('cell_count_calibration_id')
                        )
                    )
                    return
                else:
                    warn("Unexpectedly found other job in queue {}".format(
                        model
                    ))
            else:
                warn("Unexpectedly found other job in queue {}".format(
                    item
                ))

    uri = (
        scanomatic +
        '/api/analysis/instructions/testproject/{}'.format(
            job_settings['output_directory'])
    )
    tries = 0
    while tries < 25:
        payload = requests.get(uri).json()
        if payload.get('instructions'):
            assert (
                payload['instructions'].get('ccc') ==
                job_settings['cell_count_calibration_id']), (
                "Job used unexpected CCC, found {}, expected {}".format(
                    payload['instructions'].get('ccc'),
                    job_settings['cell_count_calibration_id']
                )
            )
            assert (
                job_settings['compilation'] in
                payload['instructions'].get('compilation')
            ), "Job used unexpected compilation file {}".format(
                payload['instructions'].get('compilation')
            )

            return
        else:
            tries += 1
            sleep(min(0.5 * tries, 10))
    assert False, "Time out waiting for results at '{}'".format(
        uri
    )


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
    elem.select_by_visible_text('Testum Testis, Test Testare')

    elem = browser.find_element_by_id('analysis-directory')
    elem.send_keys('test_ccc_{}'.format(browser_name))

    elem = browser.find_element_by_css_selector(
        'label[for=chain-analysis-request]'
    )
    elem.click()

    browser.find_element_by_id('submit-button').click()

    assert_has_job(scanomatic, {
        'compilation': 'testproject/testproject.project.compilation',
        'cell_count_calibration_id': 'TESTUMz',
        'output_directory': 'test_ccc_{}'.format(browser_name),
    })
