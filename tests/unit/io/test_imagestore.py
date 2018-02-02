from __future__ import absolute_import
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
def imagestore(tmpdir):
    return ImageStore(str(tmpdir))


class TestPutImage:
    def test_file_content(self, imagestore, tmpdir, scan):
        scanjobdir = tmpdir.mkdir('bbbb')
        imagestore.put(b'I am an image', scan)
        files = scanjobdir.listdir()
        assert len(files) == 1
        assert files[0].read() == 'I am an image'

    def test_file_name(self, imagestore, tmpdir, scan):
        scanjobdir = tmpdir.mkdir('bbbb')
        imagepath = scanjobdir.join('aaaa.tiff')
        imagestore.put(b'I am an image', scan)
        assert scanjobdir.listdir() == [imagepath]

    def test_create_directory(self, imagestore, tmpdir, scan):
        imagestore.put(b'I am an image', scan)
        assert tmpdir.join('bbbb').check(dir=True)


class TestGetImage:
    def test_existing_image(self, imagestore, tmpdir, scan):
        imagepath = tmpdir.mkdir('bbbb').join('aaaa.tiff')
        with imagepath.open('wb') as f:
            f.write(b'I am an image')
        image = imagestore.get(scan)
        assert image == b'I am an image'

    def test_non_existing_image(self, imagestore, tmpdir, scan):
        with pytest.raises(ImageNotFoundError):
            imagestore.get(scan)
