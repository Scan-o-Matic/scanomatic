from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class ProjectsPage(object):
    path = '/projects'

    page_heading_locator = (By.TAG_NAME, 'h1')
    add_project_button_locator = (By.CSS_SELECTOR, 'button.new-project')

    def __init__(self, driver, baseurl):
        self.driver = driver
        self.baseurl = baseurl
        self.driver.get(self.baseurl + self.path)
        WebDriverWait(
            self.driver, 5
        ).until(EC.presence_of_element_located(self.page_heading_locator))

    def get_page_heading(self):
        element = self.driver.find_element(*self.page_heading_locator)
        return element.text

    def click_add_project(self):
        button = self.driver.find_element(*self.add_project_button_locator)
        button.click()
        return NewProjectForm(self.driver)

    def get_project_panel(self, projectname):
        return ProjectPanel(self.driver, projectname)


class NewProjectForm(object):
    form_locator = (By.CSS_SELECTOR, '.new-project-panel form')
    name_input_locator = (By.CSS_SELECTOR, 'input.name')
    description_input_locator = (By.CSS_SELECTOR, 'textarea.description')
    submit_button_locator = (By.CSS_SELECTOR, 'button[type="submit"]')

    def __init__(self, driver):
        self.driver = driver
        self.element = driver.find_element(*self.form_locator)

    def set_name(self, name):
        self.element.find_element(*self.name_input_locator).send_keys(name)

    def set_description(self, description):
        self.element.find_element(*self.description_input_locator,
                                 ).send_keys(description)

    def click_submit(self):
        self.element.find_element(*self.submit_button_locator).click()


class ProjectPanel(object):

    panel_heading_locator = (By.CLASS_NAME, 'panel-heading')
    panel_body_locator = (By.CLASS_NAME, 'panel-body')

    def project_panel_locator(self, name):
        return (
            By.CSS_SELECTOR,
            'div.project-listing[data-projectname="{}"]'.format(name)
        )

    def __init__(self, driver, name):
        self.driver = driver
        self.element = self.driver.find_element(
            *self.project_panel_locator(name)
        )

    @property
    def heading(self):
        return self.element.find_element(*self.panel_heading_locator).text

    @property
    def body(self):
        return self.element.find_element(*self.panel_body_locator).text


def test_page_is_up(scanomatic, browser):
    page = ProjectsPage(browser, scanomatic)
    assert page.get_page_heading() == 'Projects'


def test_create_project(scanomatic, browser):
    page = ProjectsPage(browser, scanomatic)
    form = page.click_add_project()
    form.set_name('My project')
    form.set_description('bla bla bla bla bla')
    form.click_submit()
    panel = page.get_project_panel('My project')
    assert panel.heading == 'My project'
    assert 'bla bla bla bla bla' in panel.body
