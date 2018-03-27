from __future__ import absolute_import

import warnings

import py.path
import pytest
import requests
from selenium import webdriver


@pytest.fixture(scope='session')
def docker_compose_file(pytestconfig):
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


def pytest_configure(config):
    scanomatic_url = config.getoption('--scanomatic-url')
    if scanomatic_url is not None:
        globals()['scanomatic'] = (
            pytest.fixture(scope='session')(lambda: scanomatic_url)
        )
    else:
        globals()['scanomatic'] = _scanomatic


@pytest.fixture()
def browser(request):
    browser = request.config.getoption('--browser')
    if browser == 'firefox':
        driver = webdriver.Firefox()
    elif browser == 'chrome':
        driver = webdriver.Chrome()
    else:
        raise ValueError('Unknown browser {}'.format(browser))
    yield driver
    driver.close()
