from urllib.parse import quote_plus
from time import sleep
from enum import Enum

from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

CurveMark = Enum('CurveMark', ['OK', 'OK_THIS', 'BAD', 'EMPTY', 'NO_GROWTH'])


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
        )


class PlatePosition(object):
    def __init__(self, elem, page):
        self.elem = elem
        self.page = page

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
        ).move_to_element(self.elem).click()


class TestQCNormPage:

    def test_title(self, scanomatic, browser):
        page = QCNormPage(browser, scanomatic)
        assert page.title == 'Quality Control'

    def test_clicking_select_a_project_shows_selector(
        self, scanomatic, browser, with_analysis
    ):
        page = QCNormPage(browser, scanomatic)
        assert not page.is_visible_project_selection
        page.toggle_select_project()
        assert page.is_visible_project_selection


class TestQCNormPagePreloadedProject:

    def test_correctly_alerts_to_missing_analysis(self, browser, scanomatic):
        page = QCNormPagePreloadedProject(
            browser, scanomatic, 'with_analysis', 'has no analysis',
        )
        assert page.has_analysis is False

    def test_can_load_analysis(self, browser, scanomatic, with_analysis):
        page = QCNormPagePreloadedProject(
            browser,
            scanomatic,
            with_analysis,
            with_analysis[-2],
        )
        assert page.has_analysis

    def test_has_correct_name(self, browser, scanomatic, with_analysis):
        page = QCNormPagePreloadedProject(
            browser,
            scanomatic,
            with_analysis,
            with_analysis[-2],
        )
        details = page.get_details()
        assert details.name == with_analysis[-2]


class TestQCNormPagePlates:
    def test_loads_first_plate(self, browser, scanomatic, with_analysis):
        page = QCNormPagePreloadedProject(
            browser,
            scanomatic,
            with_analysis,
            with_analysis[-2],
        )
        plate = page.get_plate_display_area()
        assert plate.number == '1'

    def test_switches_plate(self, browser, scanomatic, with_analysis):
        page = QCNormPagePreloadedProject(
            browser,
            scanomatic,
            with_analysis,
            with_analysis[-2],
        )
        plate = page.get_plate_display_area()
        page.set_plate(3)
        assert plate.number == '3'


class TestQCNormCurveMarking:
    def test_action_buttons_can_toggle_visibility(
        self, browser, scanomatic, with_analysis
    ):
        page = QCNormPagePreloadedProject(
            browser,
            scanomatic,
            with_analysis,
            with_analysis[-2],
        )
        plate = page.get_plate_display_area()
        plate.toggle_mark_experiments()
        visible = plate.has_marking_enabled
        plate.toggle_mark_experiments()
        assert plate.has_marking_enabled != visible

    def test_mark_current_curve_as_ok(
        self, browser, scanomatic, with_analysis
    ):
        page = QCNormPagePreloadedProject(
            browser,
            scanomatic,
            with_analysis,
            with_analysis[-2],
        )
        plate = page.get_plate_display_area()
        plate.show_mark_experiments()
        plate.mark_selected_curve(CurveMark.OK)

    def test_marking_curve_updates_plate_view(
        self, browser, scanomatic, with_analysis
    ):
        page = QCNormPagePreloadedProject(
            browser,
            scanomatic,
            with_analysis,
            with_analysis[-2],
        )
        plate = page.get_plate_display_area()
        plate.show_mark_experiments()
        position = plate.get_plate_position(5, 10)
        assert position.mark == CurveMark.OK
        position.click()
        position = plate.get_plate_position(5, 10)
        assert position.mark == CurveMark.BAD
