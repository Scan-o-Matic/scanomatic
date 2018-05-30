from __future__ import absolute_import

import os
import shutil
import uuid

import py.path
import pytest
import requests
from selenium import webdriver
from selenium.common.exceptions import WebDriverException


@pytest.fixture(scope='session')
def docker_mounts_preparation():
    path = os.path.join('/', 'tmp', 'som-analysis-testdata')
    os.makedirs(path)
    yield
    shutil.rmtree(path)


@pytest.fixture(scope='session')
def docker_compose_file(docker_mounts_preparation, pytestconfig):
    return [
        pytestconfig.rootdir.join('docker-compose.yml'),
        py.path.local(__file__).dirpath().join('docker-compose.override.yml'),
    ]


@pytest.fixture(scope='session')
def _scanomatic(docker_ip, docker_services):

    def is_responsive(url):
        try:
            requests.get(url).raise_for_status()
        except requests.RequestException:
            return False
        else:
            return True

    url = 'http://{}:{}'.format(
        docker_ip,
        docker_services.port_for('scanomatic-frontend', 5000),
    )
    docker_services.wait_until_responsive(
        timeout=30, pause=0.1,
        check=lambda: is_responsive(url + '/fixtures')
    )
    return url


@pytest.fixture
def chrome():
    driver = webdriver.Chrome()
    yield driver
    driver.quit()


@pytest.fixture
def firefox():
    driver = webdriver.Firefox()
    yield driver
    driver.quit()


SAUCELABS_BROWSERS = [
    {
        'browserName': 'firefox',
        'platform': 'Windows',
        'version': '59'
    },
    {
        'browserName': 'chrome',
        'platform': 'Windows',
        'version': '65'
    },
]


@pytest.fixture(
    ids=[
        '{browserName} {version} on {platform}'.format(**browser)
        for browser in SAUCELABS_BROWSERS
    ],
    params=SAUCELABS_BROWSERS,
)
def saucelabs(request):
    username = os.environ['SAUCE_USERNAME']
    access_key = os.environ['SAUCE_ACCESS_KEY']
    build_number = os.environ['TRAVIS_BUILD_NUMBER']
    job_number = os.environ['TRAVIS_JOB_NUMBER']
    repo_slug = os.environ['TRAVIS_REPO_SLUG']
    capabilities = dict(request.param)
    capabilities['tunnel-identifier'] = job_number
    capabilities['build'] = '{}#{}'.format(repo_slug, build_number)
    capabilities['name'] = request.node.nodeid
    hub_url = '%s:%s@localhost:4445' % (username, access_key)
    driver = webdriver.Remote(
        desired_capabilities=capabilities,
        command_executor='http://%s/wd/hub' % hub_url,
    )
    yield driver
    try:
        driver.execute_script(
            'sauce:job-result={}'.format(request.node.call_result.outcome)
        )
        driver.quit()
    except WebDriverException:
        print('Warning: The driver failed to quit properly.')


def pytest_configure(config):
    browser = config.getoption('--browser')
    globals()['browser'] = globals()[browser]
    scanomatic_url = config.getoption('--scanomatic-url')
    if scanomatic_url is not None:
        globals()['scanomatic'] = (
            pytest.fixture(scope='session')(lambda: scanomatic_url)
        )
    else:
        globals()['scanomatic'] = _scanomatic


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # this sets the result as a test attribute for SauceLabs reporting.
    # execute all other hooks to obtain the report object
    outcome = yield
    result = outcome.get_result()
    # set an report attribute for each phase of a call, which can
    # be "setup", "call", "teardown"
    setattr(item, "{}_result".format(result.when), result)


@pytest.fixture(scope='function')
def experiment_only_analysis():
    project = str(uuid.uuid4()).replace('-', '')
    shutil.copytree(
        os.path.join(os.path.dirname(__file__), 'data', 'analysis'),
        os.path.join('/', 'tmp', 'som-analysis-testdata', project, 'analysis'),
    )
    return ['experiments_only_analysis', project, 'analysis']
