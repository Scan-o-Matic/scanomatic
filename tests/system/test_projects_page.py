from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class ProjectsPage(object):
    path = '/projects'

    page_heading_locator = (By.TAG_NAME, 'h1')

    def __init__(self, driver, baseurl):
        self.driver = driver
        self.baseurl = baseurl
        self.driver.get(self.baseurl + self.path)
        WebDriverWait(
            self.driver, 10
        ).until(EC.presence_of_element_located(self.page_heading_locator))

    def get_page_heading(self):
        element = self.driver.find_element(*self.page_heading_locator)
        return element.text


def test_page_is_up(scanomatic, browser):
    page = ProjectsPage(browser, scanomatic)
    assert page.get_page_heading() == 'Projects'
