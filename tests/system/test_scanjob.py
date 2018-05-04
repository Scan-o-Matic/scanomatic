from __future__ import absolute_import

import uuid
import requests
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep


def test_compile_with_given_directory(scanomatic, browser):
    scanjobname = 'Scan job {}'.format(uuid.uuid4().hex)

    scannerid = create_scanner(scanomatic)
    scanjobid = create_scanjob(scanomatic, scannerid, scanjobname)
    start_scanjob(scanomatic, scanjobid)
    post_scan(scanomatic, scanjobid)

    sleep(2)
    browser.get(scanomatic + '/experiment')
    browser.find_element_by_id('job-' + scanjobid).find_element_by_link_text(
        "Compile".format(scanjobname)
    ).click()
    input = browser.find_element_by_id('project-directory')
    assert input.get_attribute('value') == 'root/{}'.format(scanjobid)
    browser.find_element_by_id('manual-selection').click()
    images = browser.find_elements_by_class_name('image-list-label')
    assert len(images) == 1


def test_qc_with_given_directory(scanomatic, browser):
    scanjobname = 'Scan job {}'.format(uuid.uuid4().hex)

    scannerid = create_scanner(scanomatic)
    scanjobid = create_scanjob(scanomatic, scannerid, scanjobname)
    start_scanjob(scanomatic, scanjobid)

    sleep(2)
    browser.get(scanomatic + '/experiment')
    browser.find_element_by_id('job-' + scanjobid).find_element_by_link_text(
        "Quality Control".format(scanjobname)
    ).click()
    WebDriverWait(browser, 2).until(
        EC.presence_of_element_located((By.ID, 'divLoading'))
    )
    modal = browser.find_element_by_id('divLoading')
    assert modal.text == 'Error: No analysis found!'


def create_scanner(scanomatic):
    scannerid = uuid.uuid4().hex
    response = requests.put(
        scanomatic + '/api/scanners/{}/status'.format(scannerid),
        json={
            'startTime': '2000-01-02T00:00:00Z',
            'imagesToSend': 0,
            'devices': ['epson'],
        },
    )
    response.raise_for_status()
    return scannerid


def create_scanjob(scanomatic, scannerid, name):
    response = requests.post(
        scanomatic + '/api/scan-jobs',
        json={
            'name': name,
            'interval': 300,
            'duration': 1,
            'scannerId': scannerid,
        },
    )
    response.raise_for_status()
    scanjobid = response.json()['identifier']
    return scanjobid


def start_scanjob(scanomatic, scanjobid):
    response = requests.post(
        scanomatic + '/api/scan-jobs/{}/start'.format(scanjobid)
    )
    response.raise_for_status()


def post_scan(scanomatic, scanjobid):
    response = requests.post(
        scanomatic + '/api/scans',
        data={
            'scanJobId': scanjobid,
            'startTime': '2000-01-02T00:05:00Z',
            'endTime': '2000-01-02T00:10:00Z',
            'digest': 'sha256:xxx',
        },
        files={
            'image': b'xxxx',
        },
    )
    response.raise_for_status()
