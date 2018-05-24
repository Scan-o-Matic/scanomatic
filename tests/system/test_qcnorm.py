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

    def load(self, *path):
        elem = self.driver.find_element(By.CSS_SELECTOR, '#selectProject')
        for p in path:
            elem = elem.find_element(
                By.XPATH,
                '//td[contains(., \'{}\')]'.format(p),
            )
            a = elem.find_element(By.TAG_NAME, 'a')
            a.click()

        if not elem.tag == 'td':
            raise AnalysisNotFoundError("Could not locate: {}".format(path))

        btn = elem.find_element(By.TAG_NAME, 'button')
        if (
            not btn
            or btn.id != '/'.join(path)
            or btn.text != 'Here is your project'
        ):
            raise AnalysisNotFoundError('No analysis for: {}'.format(path))
        btn.click()


class TestQCNormPage:

    def test_title(self, scanomatic, browser):
        page = QCNormPage(browser, scanomatic)
        assert page.title == 'Quality Control'

    def test_clicking_select_a_project_shows_selector(
        self, scanomatic, browser,
    ):
        page = QCNormPage(browser, scanomatic)
        assert not page.is_visible_project_selection
        page.toggle_select_project()
        assert page.is_visible_project_selection

    def test_can_load_analysis(self, browser, scanomatic, with_analysis):
        page = QCNormPage(browser, scanomatic)
        page.toggle_select_project()
        page.load(*with_analysis)
