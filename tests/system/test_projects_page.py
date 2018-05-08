from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait


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
    errors_locator = (By.CSS_SELECTOR, '.form-group.has-error .help-block')

    def __init__(self, driver):
        self.driver = driver
        self.element = driver.find_element(*self.form_locator)

    def set_name(self, name):
        self.element.find_element(*self.name_input_locator).send_keys(name)

    def set_description(self, description):
        input = self.element.find_element(*self.description_input_locator)
        input.send_keys(description)

    @property
    def errors(self):
        return [
            el.text for el in self.element.find_elements(*self.errors_locator)
        ]

    def click_submit(self):
        self.element.find_element(*self.submit_button_locator).click()


class ProjectPanel(object):

    panel_heading_locator = (By.CLASS_NAME, 'panel-heading')
    panel_body_locator = (By.CLASS_NAME, 'panel-body')
    add_experiment_button_locator = (By.CSS_SELECTOR, 'button.new-experiment')

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

    def click_add_experiment(self):
        button = self.driver.find_element(*self.add_experiment_button_locator)
        button.click()
        return NewExperimentForm(self.element)

    def get_experiment_panel(self, name):
        return ExperimentPanel(self.element, name)

    def toggle_expanded(self):
        return self.element.find_element(*self.panel_heading_locator).click()


class NewExperimentForm(object):
    form_locator = (By.CSS_SELECTOR, '.new-experiment-panel')
    name_input_locator = (By.CSS_SELECTOR, 'input.name')
    description_input_locator = (By.CSS_SELECTOR, 'textarea.description')
    interval_input_locator = (By.CSS_SELECTOR, 'input.interval')
    scanner_select_locator = (By.CSS_SELECTOR, 'select.scanner')
    submit_button_locator = (By.CSS_SELECTOR, 'button.experiment-add')
    errors_locator = (By.CSS_SELECTOR, '.form-group.has-error .help-block')

    @staticmethod
    def duration_input_locator(unit):
        return (By.CSS_SELECTOR, 'input.{}'.format(unit))

    def __init__(self, parent):
        self.driver = parent
        self.element = parent.find_element(*self.form_locator)

    def set_name(self, name):
        self.element.find_element(*self.name_input_locator).send_keys(name)

    def set_description(self, description):
        input = self.element.find_element(*self.description_input_locator)
        input.send_keys(description)

    def set_duration(self, days, hours, minutes):
        days_input = self.element.find_element(
            *self.duration_input_locator('days')
        )
        days_input.send_keys(days)
        hours_input = self.element.find_element(
            *self.duration_input_locator('hours')
        )
        hours_input.send_keys(hours)
        minutes_input = self.element.find_element(
            *self.duration_input_locator('minutes')
        )
        minutes_input.send_keys(minutes)

    def set_interval(self, minutes):
        input = self.element.find_element(*self.interval_input_locator)
        input.send_keys(minutes)

    def set_scanner(self, scannerid):
        element = self.element.find_element(*self.scanner_select_locator)
        select = Select(element)
        select.select_by_value(scannerid)

    def click_submit(self):
        self.element.find_element(*self.submit_button_locator).click()

    @property
    def errors(self):
        return [
            el.text for el in self.element.find_elements(*self.errors_locator)
        ]


class ExperimentPanel(object):

    panel_heading_locator = (By.CLASS_NAME, 'panel-heading')
    panel_body_locator = (By.CLASS_NAME, 'panel-body')

    @staticmethod
    def experiment_panel_locator(name):
        return (
            By.CSS_SELECTOR,
            'div.experiment-listing[data-experimentname="{}"]'.format(name)
        )

    def __init__(self, parent, name):
        self.element = parent.find_element(
            *self.experiment_panel_locator(name)
        )

    @property
    def heading(self):
        return self.element.find_element(*self.panel_heading_locator).text

    @property
    def body(self):
        return self.element.find_element(*self.panel_body_locator).text

    @property
    def stats(self):
        trs = self.element.find_element_by_class_name(
            'experiment-stats'
        ).find_elements_by_tag_name('tr')
        return {
            tds[0].text: tds[1].text
            for tds in [tr.find_elements_by_tag_name('td') for tr in trs]
        }

    def toggle_expanded(self):
        return self.element.find_element(*self.panel_heading_locator).click()


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
    panel.toggle_expanded()
    assert 'bla bla bla bla bla' in panel.body


def test_create_project_without_name(scanomatic, browser):
    page = ProjectsPage(browser, scanomatic)
    form = page.click_add_project()
    form.set_name('')
    form.set_description('bla bla bla bla bla')
    form.click_submit()
    assert 'Project name cannot be empty' in form.errors


def test_create_experiment(scanomatic, browser):
    page = ProjectsPage(browser, scanomatic)
    form = page.click_add_project()
    form.set_name('My project')
    form.set_description('bla bla bla bla bla')
    form.click_submit()
    project_panel = page.get_project_panel('My project')
    project_panel.toggle_expanded()
    form = project_panel.click_add_experiment()
    form.set_name('My Experiment')
    form.set_description('Lorem ipsum dolor sit amet')
    form.set_duration(days=1, hours=2, minutes=3)
    form.set_interval(4)
    form.set_scanner('0000')
    form.click_submit()
    experiment_panel = project_panel.get_experiment_panel('My Experiment')
    experiment_panel.toggle_expanded()
    assert 'My Experiment' in experiment_panel.heading
    assert 'Lorem ipsum dolor sit amet' in experiment_panel.body
    assert '1 days 2 hours 3 minutes' in experiment_panel.stats['Duration']
    assert '4 minutes' in experiment_panel.stats['Interval']
    assert 'Scanner One' in experiment_panel.stats['Scanner']


def test_create_experiment_without_name(scanomatic, browser):
    page = ProjectsPage(browser, scanomatic)
    form = page.click_add_project()
    form.set_name('My project')
    form.set_description('bla bla bla bla bla')
    form.click_submit()
    project_panel = page.get_project_panel('My project')
    project_panel.toggle_expanded()
    form = project_panel.click_add_experiment()
    form.set_name('')
    form.set_description('Lorem ipsum dolor sit amet')
    form.set_duration(days=1, hours=2, minutes=3)
    form.set_interval(123)
    form.set_scanner('0000')
    form.click_submit()
    assert 'Required' in form.errors
