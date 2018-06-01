from urllib.parse import quote_plus
from enum import Enum

import pytest
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

CurveMark = Enum('CurveMark', ['OK', 'OK_THIS', 'BAD', 'EMPTY', 'NO_GROWTH'])
Navigations = Enum('Navigations', ['NEXT', 'PREV', 'RESET'])


class AnalysisNotFoundError(Exception):
    pass


class QCNormPageBase(object):
    TALKING_TO_SERVER_TIMEOUT = 10
    PROJECT_LOAD_CHECKS_TIMEOUT = 60
    details_selector = (By.CSS_SELECTOR, '#tbProjectDetails')

    def __init__(self, driver, base_url, path):
        self.driver = driver
        self.base_url = base_url
        driver.get(base_url + path)

    @property
    def is_visible_project_selection(self):
        return (
            self.driver.find_element(By.CSS_SELECTOR, '#selectProject')
                .value_of_css_property('height') not in ['0px', 'auto']
        )

    @property
    def title(self):
        return self.driver.find_element(By.TAG_NAME, 'h1').text

    def wait_until_not_talking_to_server(self):
        loading_selector = (
            By.XPATH,
            "//*[@id='divLoading' and contains(., 'Talking to server')]",
        )

        WebDriverWait(self.driver, self.TALKING_TO_SERVER_TIMEOUT).until_not(
            EC.visibility_of_element_located(loading_selector),
        )

    def wait_until_project_loaded(self):
        WebDriverWait(self.driver, self.PROJECT_LOAD_CHECKS_TIMEOUT).until(
            EC.visibility_of_element_located(self.details_selector),
        )
        WebDriverWait(self.driver, self.PROJECT_LOAD_CHECKS_TIMEOUT).until(
            EC.visibility_of_element_located(
                self._get_plate_btn_selector(1),
            )
        )
        self.wait_until_not_talking_to_server()

    def get_details(self):
        self.wait_until_project_loaded()
        return ProjectDetails(self.driver.find_element(*self.details_selector))

    def set_plate(self, plate):
        """Setting plate by number as shown in UI (1 = first)
        """
        self.driver.find_element(*self._get_plate_btn_selector(plate)).click()
        self.wait_until_not_talking_to_server()

    def get_plate_display_area(self):
        self.wait_until_project_loaded()
        return PlateDisplayArea(
            self.driver.find_element(By.CSS_SELECTOR, '#displayArea'),
            self,
        )

    def set_phenotype(self, phenotype_name):
        self.wait_until_project_loaded()
        id = '#selRunPhenotypes'
        select = Select(self.driver.find_element(By.CSS_SELECTOR, id))
        select.select_by_visible_text(phenotype_name)
        self.wait_until_not_talking_to_server()


class QCNormPage(QCNormPageBase):
    path = '/qc_norm'

    def __init__(self, driver, base_url):
        super().__init__(driver, base_url, self.path)

    def toggle_select_project(self):
        self.driver.find_element(By.CSS_SELECTOR, '#btnBrowseProject').click()


class QCNormPagePreloadedProject(QCNormPageBase):
    path = '/qc_norm?analysisdirectory={}&project={}'

    def __init__(self, driver, base_url, project_path, project_name):
        path = self.path.format(
            quote_plus('/'.join(project_path)),
            quote_plus(project_name),
        )
        super().__init__(driver, base_url, path)

    def _get_plate_btn_selector(self, button):
        return (By.CSS_SELECTOR, '#btnPlate{}'.format(button - 1))

    @property
    def has_analysis(self):
        try:
            return (
                self.driver.find_element(By.CSS_SELECTOR, '#divLoading')
                    .text != 'Error: No analysis found!'
            )
        except NoSuchElementException:
            return True


class ProjectDetails(object):

    def __init__(self, elem):
        self.elem = elem

    @property
    def name(self):
        return self.elem.find_element(By.CSS_SELECTOR, '#spProject_name').text


class PlateDisplayArea(object):
    mark_experiments_toggle_selector = (
        By.CSS_SELECTOR, '.mark-experiments-toggle',
    )
    mark_states_action_group_selector = (
        By.CSS_SELECTOR, '.mark-experiments-action-group',
    )

    def __init__(self, elem, page):
        self.elem = elem
        self.page = page

    @property
    def number(self):
        return self.elem.find_element(By.CSS_SELECTOR, '#spnPlateIdx').text

    @property
    def has_marking_enabled(self):
        return self.elem.find_element(
            *self.mark_states_action_group_selector).is_displayed()

    def toggle_mark_experiments(self):
        span = self.elem.find_element(*self.mark_experiments_toggle_selector)
        span.find_element(By.CSS_SELECTOR, '.toggle').click()

    def show_mark_experiments(self):
        self.toggle_mark_experiments()
        while not self.has_marking_enabled:
            self.toggle_mark_experiments()

    def mark_selected_curve(self, mark):
        id = ''
        if mark == CurveMark.OK:
            id = '#btnMarkOK'
        elif mark == CurveMark.OK_THIS:
            id = '#btnMarkOKOne'
        elif mark == CurveMark.BAD:
            id = '#btnMarkBad'
        elif mark == CurveMark.EMPTY:
            id = '#btnMarkEmpty'
        elif mark == CurveMark.NO_GROWTH:
            id = '#btnMarkNoGrowth'
        self.elem.find_element(By.CSS_SELECTOR, id).click()
        self.page.wait_until_not_talking_to_server()

    def get_plate_position(self, row, col):
        """Positions from upper left corner as is_displayed

        Starting index 1
        """
        id = '#id{}_{}'.format(row - 1, col - 1)
        return PlatePosition(
            self.elem.find_element(By.CSS_SELECTOR, id),
            self.page,
            row,
            col,
        )

    def get_qindex(self):
        id = "#qIndexCurrent"
        return self.elem.find_element(By.CSS_SELECTOR, id).text()

    def update_qindex(self, operation):
        id = ''
        if operation == CurveMark.NEXT:
            id = '#btnQidxNext'
        elif operation == CurveMark.PREV:
            id = '#btnQidxPrev'
        elif operation == CurveMark.RESET:
            id = '#btnQidxReset'
        self.elem.find_element(By.CSS_SELECTOR, id).click()


class PlatePosition(object):
    def __init__(self, elem, page, row, col):
        self.elem = elem
        self.page = page
        self.row = row - 1
        self.col = col - 1

    @property
    def mark(self):
        mark = self.elem.get_attribute('data-meta-type')
        if mark == 'OK':
            return CurveMark.OK
        elif mark == 'BadData':
            return CurveMark.BAD
        elif mark == 'Empty':
            return CurveMark.EMPTY
        elif mark == 'NoGrowth':
            return CurveMark.NO_GROWTH

    def click(self):
        ActionChains(
            self.page.driver,
        ).move_to_element(self.elem).click().perform()
        WebDriverWait(self.page.driver, 5).until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    "//*[@id='sel' and contains(., '[{},{}]')]".format(
                        self.row,
                        self.col,
                    ),
                )
            )
        )


class TestQCNormPage:

    def test_title(self, scanomatic, browser):
        page = QCNormPage(browser, scanomatic)
        assert page.title == 'Quality Control'

    def test_clicking_select_a_project_shows_selector(
        self, scanomatic, browser, experiment_only_analysis
    ):
        page = QCNormPage(browser, scanomatic)
        assert not page.is_visible_project_selection
        page.toggle_select_project()
        assert page.is_visible_project_selection


class TestQCNormPagePreloadedProject:

    def test_correctly_alerts_to_missing_analysis(self, browser, scanomatic):
        page = QCNormPagePreloadedProject(
            browser,
            scanomatic,
            'experiments_only_analysis',
            'has no analysis',
        )
        assert page.has_analysis is False

    def test_can_load_analysis(
        self, browser, scanomatic, experiment_only_analysis
    ):
        page = QCNormPagePreloadedProject(
            browser,
            scanomatic,
            experiment_only_analysis,
            experiment_only_analysis[-2],
        )
        assert page.has_analysis
        details = page.get_details()
        assert details.name == experiment_only_analysis[-2]


class TestQCNormPagePlates:

    def test_switches_plate(
        self, browser, scanomatic, experiment_only_analysis
    ):
        page = QCNormPagePreloadedProject(
            browser,
            scanomatic,
            experiment_only_analysis,
            experiment_only_analysis[-2],
        )
        plate = page.get_plate_display_area()
        page.set_plate(3)
        assert plate.number == '3'


class TestQCNormCurveMarking:

    def test_marking_ok_this_only_ok_current_phenotype(
        self, browser, scanomatic, experiment_only_analysis
    ):
        pos = (5, 10)
        page = QCNormPagePreloadedProject(
            browser,
            scanomatic,
            experiment_only_analysis,
            experiment_only_analysis[-2],
        )
        plate = page.get_plate_display_area()
        plate.show_mark_experiments()
        position = plate.get_plate_position(*pos)
        assert position.mark == CurveMark.OK
        position.click()
        plate.mark_selected_curve(CurveMark.BAD)
        page.set_phenotype('Experiment Growth Yield')
        position = plate.get_plate_position(*pos)
        assert position.mark == CurveMark.BAD
        position.click()
        plate.mark_selected_curve(CurveMark.OK_THIS)
        position = plate.get_plate_position(*pos)
        assert position.mark == CurveMark.OK
        page.set_phenotype('Generation Time')
        position = plate.get_plate_position(*pos)
        assert position.mark == CurveMark.BAD


class TestQCNormNavigateQidx:

    @classmethod
    @pytest.fixture(scope='function')
    def plate_page(browser, scanomatic, experiment_only_analysis):
        page = QCNormPagePreloadedProject(
            browser,
            scanomatic,
            experiment_only_analysis,
            experiment_only_analysis[-2],
        )
        return page

    def test_page_starts_on_lowest_qindex(self, plate_page):
        assert plate_page.get_qindex == "1"

    def test_pressing_btns_changes_qindex(self, plate_page):
        plate_page.update_qindex(Navigations.NEXT)
        assert plate_page.get_qindex == "2"
        plate_page.update_qindex(Navigations.PREV)
        assert plate_page.get_qindex == "1"

    def test_qindex_overflow_wraps(self, plate_page):
        plate_page.update_qindex(Navigations.PREV)
        assert plate_page.get_qindex == "1536"
        plate_page.update_qindex(Navigations.NEXT)
        assert plate_page.get_qindex == "1"

    def test_pressing_reset_goes_to_first_qindex(self, plate_page):
        plate_page.update_qindex(Navigations.NEXT)
        plate_page.update_qindex(Navigations.NEXT)
        assert plate_page.get_qindex == "3"
        plate_page.update_qindex(Navigations.RESET)
        assert plate_page.get_qindex == "1"
