import pytest
import requests
from selenium import webdriver


@pytest.fixture(scope='session')
def docker_compose_file(pytestconfig):
    return (
        pytestconfig.rootdir.join('docker-compose.yml'),
        pytestconfig.rootdir.join('docker-compose.override.yml'),
    )


@pytest.fixture(scope='session')
def scanomatic(docker_ip, docker_services):
    ### return "http://localhost:5001"
    def is_responsive(url):
        try:
            requests.get(url).raise_for_status()
        except requests.RequestException:
            return False
        else:
            return True

    url = 'http://{}:{}'.format(
        docker_ip,
        docker_services.port_for('scanomatic', 5000),
    )
    docker_services.wait_until_responsive(
        timeout=30, pause=0.1,
        check=lambda: is_responsive(url + '/fixtures')
    )
    return url


@pytest.fixture(
    'session',
    ids=['chrome', 'firefox'],
    params=[webdriver.Chrome, webdriver.Firefox],
)
def browser(request):
    driver = request.param()
    yield driver
    driver.close()
