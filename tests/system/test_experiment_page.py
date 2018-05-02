from uuid import uuid4

import pytest
import requests
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait


def create_scanner(scanomatic, scannerid):
    response = requests.put(
        scanomatic + '/api/scanners/{}/status'.format(scannerid),
        json={
            'startTime': '2000-01-02T00:00:00Z',
            'imagesToSend': 0,
            'devices': ['epson'],
        },
    )
    response.raise_for_status()


@pytest.fixture(autouse=True, scope='module')
def scanner(scanomatic):
    scannerid = uuid4().hex
    create_scanner(scanomatic, scannerid)
    return scannerid


class ExperimentPage(object):
    path = '/experiment'

    scanner_select_locator = (By.CSS_SELECTOR, 'select.scanner')

    def scanning_job_panel_locator(self, name):
        return (
            By.CSS_SELECTOR, 'div.job-listing[data-jobname="{}"]'.format(name)
        )

    def __init__(self, driver, baseurl):
        self.driver = driver
        self.baseurl = baseurl
        self.driver.get(self.baseurl + self.path)

    def add_job(self, name, scannerid=None):
        btn = self.driver.find_element_by_css_selector('button.new-job')
        btn.click()
        form = self.driver.find_element_by_css_selector('div.panel-body')
        name_input = form.find_element_by_css_selector('input.name')
        name_input.send_keys(name)
        scanner_select = self.driver.find_element(*self.scanner_select_locator)
        WebDriverWait(
            scanner_select, 5
        ).until(EC.presence_of_element_located((By.TAG_NAME, 'option')))
        if scannerid:
            Select(scanner_select).select_by_value(scannerid)
        form.find_element_by_css_selector('button.job-add').click()
        WebDriverWait(self.driver, 5).until(
            EC.
            presence_of_element_located(self.scanning_job_panel_locator(name))
        )

    def find_job_panel_by_name(self, name):
        return ScanningJobPanel(
            self.driver.find_element(*self.scanning_job_panel_locator(name)),
            self.driver
        )

    def has_job_with_name(self, name):
        try:
            self.driver.find_element(*self.scanning_job_panel_locator(name))
            return True
        except NoSuchElementException:
            return False


class ScanningJobPanel(object):
    job_info_locator = (By.CLASS_NAME, 'job-info')
    remove_button_locator = (By.CLASS_NAME, 'scanning-job-remove-button')
    stop_button_locator = (By.CLASS_NAME, 'scanning-job-stop-button')
    stop_reason_input_locator = (
        By.CSS_SELECTOR, '.scanning-job-stop-dialogue .reason-input'
    )
    status_label_locator = (By.CLASS_NAME, 'scanning-job-status-label')
    extract_features_dialgoue_button_locator = (
        By.CLASS_NAME, 'experiment-extract-features',
    )
    extract_features_ok_locator = (By.CLASS_NAME, 'feature-extract-button')
    extract_features_cancel_locator = (By.CLASS_NAME, 'cancel-button')
    keep_qc_locator = (By.CLASS_NAME, 'keep-qc')

    def __init__(self, element, driver):
        self.element = element
        self.driver = driver

    def can_remove_job(self):
        try:
            self.element.find_element(*self.remove_button_locator)
            return True
        except NoSuchElementException:
            return False

    def remove_job(self, confirm=True):
        self.element.find_element(*self.remove_button_locator).click()
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
        WebDriverWait(self.element, 2).until(
            EC.
            text_to_be_present_in_element(self.status_label_locator, 'Running')
        )

    def can_stop_job(self):
        try:
            self.element.find_element(*self.stop_button_locator)
            return True
        except NoSuchElementException:
            return False

    def stop_job(self, reason='', confirm=True):
        self.element.find_element(*self.stop_button_locator).click()
        (
            self.element.find_element(*self.stop_reason_input_locator)
            .send_keys(reason)
        )
        if confirm:
            self.element.find_element_by_css_selector(
                '.scanning-job-stop-dialogue .confirm-button'
            ).click()
            WebDriverWait(self.element, 5).until(
                EC.text_to_be_present_in_element(
                    self.status_label_locator, 'Completed'
                )
            )
        else:
            self.element.find_element_by_css_selector(
                '.scanning-job-stop-dialogue .cancel-button'
            ).click()

    def extract_features(self, confirm=True, keep_qc=False):
        (
            self.element
                .find_element(*self.extract_features_dialgoue_button_locator)
                .click()
        )
        if keep_qc:
            self.element.find_element(*self.keep_qc_locator).click()
        if confirm:
            (
                self.element
                    .find_element(*self.extract_features_ok_locator)
                    .click()
            )
            WebDriverWait(self.element, 5).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'alert'))
            )
        else:
            (
                self.element
                    .find_element(*self.extract_features_cancel_locator)
                    .click()
            )

    def get_job_stats(self):
        info = {}
        for row in self.element.find_elements(*self.job_info_locator):
            tds = row.find_elements_by_tag_name('td')
            info[tds[0].text] = tds[1].text
        return info

    def get_job_status(self):
        return self.element.find_element(*self.status_label_locator).text

    def get_warning_alert(self):
        return self.element.find_element(By.CLASS_NAME, 'alert-danger').text


class TestRemoveJob:

    def test_remove_scanning_job(self, scanomatic, browser):
        page = ExperimentPage(browser, scanomatic)
        name = 'Test job that should be removed {}'.format(uuid4())
        page.add_job(name=name)
        panel = page.find_job_panel_by_name(name)
        panel.remove_job(confirm=True)
        assert not page.has_job_with_name(name)

    def test_cancel_removing_scanning_job(self, scanomatic, browser):
        page = ExperimentPage(browser, scanomatic)
        name = 'Test job that should not be removed {}'.format(uuid4())
        page.add_job(name=name)
        panel = page.find_job_panel_by_name(name)
        panel.remove_job(confirm=False)
        assert page.has_job_with_name(name)

    def test_cannot_remove_started_job(self, scanomatic, browser):
        scannerid = uuid4().hex
        create_scanner(scanomatic, scannerid)
        page = ExperimentPage(browser, scanomatic)
        name = 'Started test job that cannot be removed {}'.format(uuid4())
        page.add_job(name=name, scannerid=scannerid)
        panel = page.find_job_panel_by_name(name)
        panel.start_job()
        assert not panel.can_remove_job()


class TestStopJob:

    def test_stop_running_job(self, scanomatic, browser):
        jobname = 'Started test job that should be stopped {}'.format(uuid4())
        scannerid = uuid4().hex
        reason = "I don't want to do this anymore!"
        create_scanner(scanomatic, scannerid)
        page = ExperimentPage(browser, scanomatic)
        page.add_job(name=jobname, scannerid=scannerid)
        panel = page.find_job_panel_by_name(jobname)
        panel.start_job()
        panel.stop_job(confirm=True, reason=reason)
        assert panel.get_job_status() == 'Completed'
        job_stats = panel.get_job_stats()
        assert 'Stopped' in job_stats
        assert job_stats['Reason'] == reason

    def test_cancel_stopping_running_job(self, scanomatic, browser):
        jobname = 'Started test job that should be stopped {}'.format(uuid4())
        scannerid = uuid4().hex
        create_scanner(scanomatic, scannerid)
        page = ExperimentPage(browser, scanomatic)
        page.add_job(name=jobname, scannerid=scannerid)
        panel = page.find_job_panel_by_name(jobname)
        panel.start_job()
        panel.stop_job(confirm=False)
        assert panel.get_job_status() == 'Running'
        assert 'Stopped' not in panel.get_job_stats()

    def test_cannot_stop_planned_job(self, scanomatic, browser):
        jobname = 'Started test job that should be stopped {}'.format(uuid4())
        page = ExperimentPage(browser, scanomatic)
        page.add_job(name=jobname)
        panel = page.find_job_panel_by_name(jobname)
        assert not panel.can_stop_job()


class TestCompletedJob:
    def test_requesting_feature_extract_on_job_without_analysis_causes_error(
        self, scanomatic, browser
    ):
        jobname = 'Started test job that should be stopped {}'.format(uuid4())
        scannerid = uuid4().hex
        reason = "I don't want to do this anymore!"
        create_scanner(scanomatic, scannerid)
        page = ExperimentPage(browser, scanomatic)
        page.add_job(name=jobname, scannerid=scannerid)
        panel = page.find_job_panel_by_name(jobname)
        panel.start_job()
        panel.stop_job(confirm=True, reason=reason)
        panel.extract_features(confirm=True, keep_qc=True)
        assert 'Extraction refused:' in panel.get_warning_alert()
