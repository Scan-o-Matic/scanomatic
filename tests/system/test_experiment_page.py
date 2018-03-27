from uuid import uuid4

import pytest
import requests
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import Select


@pytest.fixture(autouse=True, scope='module')
def scanner(scanomatic):
    scannerid = uuid4().hex
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


class PageObject(object):
    path = '/'

    def __init__(self, driver, baseurl):
        self.driver = driver
        self.baseurl = baseurl
        self.driver.get(self.baseurl + self.path)


class ExperimentPage(PageObject):
    path = '/experiment'

    def add_job(self, name, scannerid=None):
        btn = self.driver.find_element_by_css_selector('button.new-job')
        btn.click()
        form = self.driver.find_element_by_css_selector('div.panel-body')
        name_input = form.find_element_by_css_selector('input.name')
        name_input.send_keys(name)
        scanner_select = Select(
            self.driver.find_element_by_css_selector('select.scanner')
        )
        if scannerid:
            scanner_select.select_by_value(scannerid)
        form.find_element_by_css_selector('button.job-add').click()

    def find_job_panel_by_name(self, name):
        for el in self.driver.find_elements_by_css_selector('div.job-listing'):
            title = el.find_element_by_tag_name('h3').text
            print('title', title)
            if title == name:
                return ExperimentPage.ScanningJobPanel(el)
        raise LookupError('No job panel with name "{}"'.format(name))

    class ScanningJobPanel(object):

        def __init__(self, element):
            self.element = element

        def can_remove_job(self):
            try:
                self.element.find_element_by_class_name(
                    'scanning-job-remove-button'
                )
                return True
            except NoSuchElementException:
                return False

        def remove_job(self, confirm=True):
            self.element.find_element_by_class_name(
                'scanning-job-remove-button'
            ).click()
            if confirm:
                self.element.find_element_by_css_selector(
                    '.scanning-job-remove-dialogue .confirm-button'
                ).click()
            else:
                self.element.find_element_by_css_selector(
                    '.scanning-job-remove-dialogue .cancel-button'
                ).click()

        def start_job(self):
            self.element.find_element_by_class_name('job-start').click()

    def has_job_with_name(self, name):
        try:
            self.find_job_panel_by_name(name)
            return True
        except LookupError:
            return False


def test_remove_scanning_job(scanomatic, browser):
    page = ExperimentPage(browser, scanomatic)
    name = 'Test job that should be removed {}'.format(uuid4())
    page.add_job(name=name)
    panel = page.find_job_panel_by_name(name)
    panel.remove_job(confirm=True)
    assert not page.has_job_with_name(name)


def test_cancel_removing_scanning_job(scanomatic, browser):
    page = ExperimentPage(browser, scanomatic)
    name = 'Test job that should not be removed {}'.format(uuid4())
    page.add_job(name=name)
    panel = page.find_job_panel_by_name(name)
    panel.remove_job(confirm=False)
    assert page.has_job_with_name(name)


def test_cannot_remove_started_job(scanomatic, browser, scanner):
    page = ExperimentPage(browser, scanomatic)
    name = 'Started test job that cannot be removed {}'.format(uuid4())
    page.add_job(name=name, scannerid=scanner)
    panel = page.find_job_panel_by_name(name)
    panel.start_job()
    assert not panel.can_remove_job()
