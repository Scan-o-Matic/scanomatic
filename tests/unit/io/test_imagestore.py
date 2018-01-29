from datetime import datetime

import pytest
from pytz import utc

from scanomatic.io.imagestore import ImageStore, ImageNotFoundError
from scanomatic.models.scan import Scan


@pytest.fixture
def scan():
    return Scan(
        id='aaaa',
        scanjob_id='bbbb',
        start_time=datetime(1985, 10, 26, 1, 20, tzinfo=utc),
        end_time=datetime(1985, 10, 26, 1, 21, tzinfo=utc),
        digest='foo:bar',
    )


@pytest.fixture
def filename():
    return 'bbbb_499137600.tiff'


@pytest.fixture
def imagestore(tmpdir):
    return ImageStore(str(tmpdir))


class TestPutImage:
    def test_directory_exists(self, imagestore, tmpdir, scan):
        scanjobdir = tmpdir.mkdir('bbbb')
        imagestore.put(b'I am an image', scan)
        imagepath = scanjobdir.join('bbbb_499137600.tiff')
        assert imagepath in scanjobdir.listdir()
        assert imagepath.read() == 'I am an image'

    def test_create_directory(self, imagestore, tmpdir, scan):
        imagestore.put(b'I am an image', scan)
        assert tmpdir.join('bbbb').check(dir=True)


class TestGetImage:
    def test_existing_image(self, imagestore, tmpdir, scan):
        imagepath = tmpdir.mkdir('bbbb').join('bbbb_499137600.tiff')
        with imagepath.open('wb') as f:
            f.write(b'I am an image')
        image = imagestore.get(scan)
        assert image == b'I am an image'

    def test_non_existing_image(self, imagestore, tmpdir, scan):
        with pytest.raises(ImageNotFoundError):
            imagestore.get(scan)
