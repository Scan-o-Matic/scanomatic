from __future__ import absolute_import

import requests


def test_root(scanomatic):
    uri = '/'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)
    assert "src='images/scan-o-matic_2.png'" in r.text


def test_home(scanomatic):
    uri = '/home'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)


def test_wiki(scanomatic):
    uri = '/wiki'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)


def test_analysis(scanomatic):
    uri = '/analysis'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)
    assert "<h1>Analysis</h1>" in r.text


def test_ccc(scanomatic):
    uri = '/ccc'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)
    assert "<h1>Cell Count Calibration</h1>" in r.text


def test_compile(scanomatic):
    uri = '/compile'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)
    assert "<h1>Compile Project</h1>" in r.text


def test_experiment(scanomatic):
    uri = '/experiment'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)
    assert '<h1>Start Scan Series</h1>' in r.text


def test_fixture(scanomatic):
    uri = '/fixtures'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)
    assert "<h1>Fixtures</h1>" in r.text


def test_help(scanomatic):
    uri = '/help'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)
    assert (
     '<h1><a id="installing">Something confusing or wrong?</a></h1>' in r.text)


def test_qc_norm(scanomatic):
    uri = '/qc_norm'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)
    assert "<h1>Quality Control</h1>" in r.text


def test_status(scanomatic):
    uri = '/status'
    r = requests.get(scanomatic + uri)
    r.raise_for_status()
    assert r.text and len(r.text), '{} is empty'.format(uri)
    assert "<h1>Status</h1>" in r.text
