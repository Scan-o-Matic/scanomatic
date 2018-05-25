from urllib.parse import quote_plus
from time import sleep

from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait


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
        )


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
    def __init__(self, elem):
        self.elem = elem

    @property
    def number(self):
        return self.elem.find_element(By.CSS_SELECTOR, '#spnPlateIdx').text


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
