import requests


def test_root(scanomatic, browser):
    uri = '/'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)
    assert "src='images/scan-o-matic_2.png'" in r.text


def test_home(scanomatic, browser):
    uri = '/home'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)


def test_wiki(scanomatic, browser):
    uri = '/wiki'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)


def test_analysis(scanomatic, browser):
    uri = '/analysis'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)
    assert "<h1>Analysis</h1>" in r.text


def test_ccc(scanomatic, browser):
    uri = '/ccc'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)
    assert "<h1>Cell Count Calibration</h1>" in r.text


def test_compile(scanomatic, browser):
    uri = '/compile'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)
    assert "<h1>Compile Project</h1>" in r.text


def test_experiment(scanomatic, browser):
    uri = '/experiment'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)
    assert '<h1 id="experiment-title">Start Experiment</h1>' in r.text


def test_feature_extract(scanomatic, browser):
    uri = '/feature_extract'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)
    assert "<h1>Feature extraction</h1>" in r.text


def test_fixture(scanomatic, browser):
    uri = '/fixtures'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)
    assert "<h1>Fixtures</h1>" in r.text


def test_help(scanomatic, browser):
    uri = '/help'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)
    assert (
     '<h1><a id="installing">Something confusing or wrong?</a></h1>' in r.text)


def test_maintain(scanomatic, browser):
    uri = '/maintain'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)
    assert "<h2>Power Manager</h2>" in r.text


def test_qc_norm(scanomatic, browser):
    uri = '/qc_norm'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)
    assert "<h1>Quality Control</h1>" in r.text


def test_status(scanomatic, browser):
    uri = '/status'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)
    assert "<h1>Status</h1>" in r.text


def test_settings(scanomatic, browser):
    uri = '/settings'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)
    assert "<h1>Settings</h1>" in r.text


def test_system_logs(scanomatic, browser):
    uri = '/logs/system/{}'

    for log in ('server', 'ui_server'):
        r = requests.get(scanomatic + uri.format(log))
        r.raise_for_status()
        assert r.text and len(r.text), '{} is empty'.format(uri)
        assert "<h1>{}</h1>".format(
            log.replace('_', ' ').capitalize(),
        ) in r.text
