from urllib.parse import quote_plus
from enum import Enum
import re

import pytest
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

UI_DEFAULT_WAIT = 20

CurveMark = Enum('CurveMark', ['OK', 'OK_THIS', 'BAD', 'EMPTY', 'NO_GROWTH'])
Navigations = Enum('Navigations', ['NEXT', 'PREV', 'SET'])


class AnalysisNotFoundError(Exception):
    pass


class QCNormPageBase(object):
    TALKING_TO_SERVER_TIMEOUT = 30
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
        graph = self.get_plate_display_area().get_graph()
        select = Select(self.driver.find_element(By.CSS_SELECTOR, id))
        select.select_by_visible_text(phenotype_name)
        self.wait_until_not_talking_to_server()
        graph.wait_until_graph_got_new_title()


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
    mark_states_action_group_selector = (
        By.CSS_SELECTOR, '.mark-experiments-action-group',
    )

    def __init__(self, elem, page):
        self.elem = elem
        self.page = page
        graph = self.get_graph()
        graph.wait_until_graph_is_visible()

    @property
    def number(self):
        return self.elem.find_element(By.CSS_SELECTOR, '#spnPlateIdx').text

    @property
    def has_marking_enabled(self):
        return self.elem.find_element(
            *self.mark_states_action_group_selector).is_displayed()

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
        return (
            self.elem.find_element(By.CSS_SELECTOR, id).get_attribute("value")
        )

    def update_qindex(self, operation):
        id = ''
        if operation == Navigations.NEXT:
            id = '#btnQidxNext'
        elif operation == Navigations.PREV:
            id = '#btnQidxPrev'
        elif operation == Navigations.SET:
            id = '#btnQidxSet'

        graph = self.get_graph()
        self.elem.find_element(By.CSS_SELECTOR, id).click()
        graph.wait_until_graph_is_updated()

    def set_qindex_input(self, keystrokes):
        id = '#qIndexCurrent'
        graph = self.get_graph()
        elem = self.elem.find_element(By.CSS_SELECTOR, id)
        elem.clear()
        elem.send_keys(keystrokes)

    def get_graph(self):
        return Graph(self)


class Graph(object):
    graph_selector = '#graph'

    def __init__(self, plate_display_area):
        self.plate_display_area = plate_display_area
        self.elem = plate_display_area.elem
        pos = self.position
        if pos:
            self.state = {
                'should_have_data': True,
                'title': self.title,
                'position': pos,
                'raw': self.get_raw_growth_data(),
                'smooth': self.get_smooth_growth_data(),
            }
        else:
            self.state = {
                'should_have_data': False,
                'title': self.title,
            }

    def __eq__(self, other):
        if (self.state['should_have_data'] != other.state['should_have_data']):
            return self.state['title'] == other.state['title']
        return (
            self.state['title'] == other.state['title']
            and self.state.get('raw') == other.state.get('raw')
            and self.state.get('smooth') == other.state.get('smooth')
        )

    def __repr__(self):
        return "<Graph '{}' Pos {} Hopefully {}, RAW {} SMOOTH {}>".format(
            self.state['title'],
            self.state.get('position'),
            'Visible' if self.state['should_have_data'] else 'Hidden',
            self.state.get('raw'),
            self.state.get('smooth'),
        )

    @property
    def title(self):
        return self.elem.find_element(By.CSS_SELECTOR, '#sel').text

    @property
    def position(self):
        pos = re.findall(r'\[([0-9]+),([0-9]+)\]', self.title)
        if not len(pos):
            return tuple()
        return tuple(map(int, pos[0]))

    def wait_until_graph_is_visible(self):
        WebDriverWait(self.elem, UI_DEFAULT_WAIT).until(
            EC.visibility_of_element_located(
                (By.CSS_SELECTOR, self.graph_selector),
            ),
        )

    def wait_until_graph_got_new_title(self):
        def got_title(*args):
            other = self.plate_display_area.get_graph()
            return self.state['title'] != other.state['title']
        WebDriverWait(None, UI_DEFAULT_WAIT).until(
            got_title,
            message='Never updated title {} vs {}'.format(
                self, self.plate_display_area.get_graph()),
        )

    def wait_until_graph_is_updated(self, *new_position):

        def has_right_position(*args):
            other = self.plate_display_area.get_graph()
            return new_position == other.state['position']
            return tuple(other.position) == new_position

        def different_data(*args):
            other = self.plate_display_area.get_graph()
            return (
                self.state.get('raw') != other.state.get('raw')
                and self.state.get('smooth') != other.state.get('smooth')
            )

        self.wait_until_graph_is_visible()
        if new_position:
            WebDriverWait(None, UI_DEFAULT_WAIT).until(
                has_right_position,
                message='Never updated to expected position {}'.format(self),
            )
            if (self.state['position'] != new_position):
                WebDriverWait(None, UI_DEFAULT_WAIT).until(
                    different_data,
                    message='Never updated plot {} vs {}'.format(
                        self, self.plate_display_area.get_graph()),
                )
        else:
            self.wait_until_graph_got_new_title()
            WebDriverWait(None, UI_DEFAULT_WAIT).until(
                different_data,
                message='Never updated plot {} vs {}'.format(
                    self, self.plate_display_area.get_graph()),
            )

    def get_raw_growth_data(self):
        graph = self.elem.find_element(By.CSS_SELECTOR, self.graph_selector)
        try:
            return graph.find_element(
                By.CSS_SELECTOR, '.raw').get_attribute('d')
        except NoSuchElementException:
            return None

    def get_smooth_growth_data(self):
        graph = self.elem.find_element(By.CSS_SELECTOR, self.graph_selector)
        try:
            return graph.find_element(
                By.CSS_SELECTOR, '.smooth').get_attribute('d')
        except NoSuchElementException:
            return None


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
        previous_graph = self.page.get_plate_display_area().get_graph()
        self.page.driver.execute_script(
            "return arguments[0].scrollIntoView(true);",
            self.elem,
        )
        ActionChains(
            self.page.driver,
        ).move_to_element(self.elem).click().perform()
        WebDriverWait(self.page.driver, UI_DEFAULT_WAIT).until(
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
        previous_graph.wait_until_graph_is_updated(self.row, self.col)


@pytest.fixture(scope='function')
def page(scanomatic, browser):
    return QCNormPage(browser, scanomatic)


@pytest.fixture(scope='function')
def page_no_analysis(browser, scanomatic):
    return QCNormPagePreloadedProject(
        browser,
        scanomatic,
        'experiments_only_analysis',
        'has no analysis',
    )


@pytest.fixture(scope='function')
def page_with_plate(browser, scanomatic, experiment_only_analysis):
    return QCNormPagePreloadedProject(
        browser,
        scanomatic,
        experiment_only_analysis,
        experiment_only_analysis[-2],
    )


class TestQCNormPage:

    def test_title(self, page):
        assert page.title == 'Quality Control'

    def test_clicking_select_a_project_shows_selector(self, page):
        assert not page.is_visible_project_selection
        page.toggle_select_project()
        assert page.is_visible_project_selection


class TestQCNormPagePreloadedProject:

    def test_correctly_alerts_to_missing_analysis(self, page_no_analysis):
        assert page_no_analysis.has_analysis is False

    def test_can_load_analysis(
            self, experiment_only_analysis, page_with_plate):
        assert page_with_plate.has_analysis
        details = page_with_plate.get_details()
        assert details.name == experiment_only_analysis[-2]


class TestQCNormPagePlates:

    def test_switches_plate(self, page_with_plate):
        plate = page_with_plate.get_plate_display_area()
        page_with_plate.set_plate(3)
        assert plate.number == '3'


class TestQCNormCurveMarking:

    def test_marking_ok_this_only_ok_current_phenotype(self, page_with_plate):
        pos = (5, 10)
        plate = page_with_plate.get_plate_display_area()
        position = plate.get_plate_position(*pos)
        assert position.mark == CurveMark.OK

        position.click()
        plate.mark_selected_curve(CurveMark.BAD)
        page_with_plate.set_phenotype('Experiment Growth Yield')
        position = plate.get_plate_position(*pos)
        assert position.mark == CurveMark.BAD

        position.click()
        plate.mark_selected_curve(CurveMark.OK_THIS)
        position = plate.get_plate_position(*pos)
        assert position.mark == CurveMark.OK

        page_with_plate.set_phenotype('Generation Time')
        position = plate.get_plate_position(*pos)
        assert position.mark == CurveMark.BAD


class TestQCNormNavigateQidx:

    def test_qindex_navigation(self, page_with_plate):
        plate = page_with_plate.get_plate_display_area()

        # Page starts at first Qindex:
        assert plate.get_qindex() == "1"
        initial_graph = plate.get_graph()

        # Pressing buttons works as expected:
        plate.update_qindex(Navigations.NEXT)
        assert plate.get_qindex() == "2"
        assert plate.get_graph() != initial_graph
        plate.update_qindex(Navigations.PREV)
        assert plate.get_qindex() == "1"
        graph_for_qindex1 = plate.get_graph()
        assert initial_graph == graph_for_qindex1

        # Qindex wraps as expexted:
        plate.update_qindex(Navigations.PREV)
        assert plate.get_qindex() == "1536"
        plate.update_qindex(Navigations.NEXT)
        assert plate.get_qindex() == "1"
        assert plate.get_graph() == graph_for_qindex1

        # Changing plate resets index:
        plate.update_qindex(Navigations.NEXT)
        assert plate.get_qindex() == "2"
        graph_plate_1 = plate.get_graph()
        page_with_plate.set_plate(3)
        assert plate.get_qindex() == "1"
        graph_plate_3 = plate.get_graph()
        assert graph_plate_1 != graph_plate_3
        assert initial_graph != graph_plate_3

        # Setting colony by clicking plate updates index:
        plate_position = plate.get_plate_position(17, 23)
        plate_position.click()
        assert plate.get_qindex() == "42"
        assert graph_plate_3 != plate.get_graph()

        # Pressing set goes back to specified index:
        plate.set_qindex_input("1")
        plate.update_qindex(Navigations.SET)
        assert plate.get_qindex() == "1"
        plate.update_qindex(Navigations.NEXT)
        assert plate.get_qindex() == "2"

        # Pressing set outside bounds goes to max or min:
        plate.set_qindex_input("-42")
        plate.update_qindex(Navigations.SET)
        assert plate.get_qindex() == "1"

        plate.set_qindex_input("1764")
        plate.update_qindex(Navigations.SET)
        assert plate.get_qindex() == "1536"
