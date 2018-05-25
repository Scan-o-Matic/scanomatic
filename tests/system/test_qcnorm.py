from urllib.parse import quote_plus

from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait


class AnalysisNotFoundError(Exception):
    pass


class QCNormPage(object):
    path = '/qc_norm'

    def __init__(self, driver, base_url):
        self.driver = driver
        self.base_url = base_url
        driver.get(base_url + self.path)

    @property
    def is_visible_project_selection(self):
        return (
            self.driver.find_element(By.CSS_SELECTOR, '#selectProject')
                .value_of_css_property('height') not in ['0px', 'auto']
        )

    @property
    def title(self):
        return self.driver.find_element(By.TAG_NAME, 'h1').text

    def toggle_select_project(self):
        self.driver.find_element(By.CSS_SELECTOR, '#btnBrowseProject').click()


class QCNormPagePreloadedProject(object):
    TALKING_TO_SERVER_TIMEOUT = 60
    path = '/qc_norm?analysisdirectory={}&project={}'

    def __init__(self, driver, base_url, project_path, project_name):
        self.driver = driver
        self.base_url = base_url
        driver.get(
            base_url +
            self.path.format(
                quote_plus('/'.join(project_path)),
                quote_plus(project_name),
            )
        )

    @property
    def has_analysis(self):
        try:
            return (
                self.driver.find_element(By.CSS_SELECTOR, '#divLoading')
                    .text != 'Error: No analysis found!'
            )
        except NoSuchElementException:
            return True

    def wait_until_loaded(self):
        loading_selector = (
            By.XPATH,
            "//*[@id='divLoading' and contains(., 'Talking to server')]",
        )

        WebDriverWait(self.driver, self.TALKING_TO_SERVER_TIMEOUT).until_not(
            EC.visibility_of_element_located(loading_selector),
        )

    def get_details(self):
        self.wait_until_loaded()
        elem_selector = (By.CSS_SELECTOR, '#tbProjectDetails')
        WebDriverWait(self.driver, 5).until(
            EC.presence_of_element_located(elem_selector),
        )
        return ProjectDetails(self.driver.find_element(*elem_selector))


class ProjectDetails(object):
    def __init__(self, elem):
        self.elem = elem

    @property
    def name(self):
        return self.elem.find_element(By.CSS_SELECTOR, '#spProject_name').text


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
