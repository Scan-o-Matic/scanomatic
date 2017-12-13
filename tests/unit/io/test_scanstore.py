from datetime import timedelta
from StringIO import StringIO

import pytest

from scanomatic.io.scanstore import ScanStore, UnknownProjectError
from scanomatic.models.scan import Scan


@pytest.fixture
def scan():
    return Scan(
        image=StringIO('I am an image'),
        index=42,
        timedelta=timedelta(seconds=1234.56789),
    )


def test_add_scan(tmpdir, scan):
    projectstore = ScanStore(str(tmpdir))
    projectdir = tmpdir.mkdir('my').mkdir('pr0ject')
    projectstore.add_scan('my/pr0ject', scan)
    imagepath = projectdir.join('pr0ject_0042_1234.5679.tiff')
    assert imagepath in projectdir.listdir()
    assert imagepath.read() == 'I am an image'


def test_add_scan_unknown_project(tmpdir, scan):
    projectstore = ScanStore(str(tmpdir))
    with pytest.raises(UnknownProjectError):
        projectstore.add_scan('unknown/pr0ject', scan)
