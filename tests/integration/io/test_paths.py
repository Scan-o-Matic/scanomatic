from __future__ import absolute_import
import pytest

from scanomatic.io.paths import make_directory


class TestMakeDirectory:
    def test_simple_directory(self, tmpdir):
        path = tmpdir.join('foo')
        make_directory(str(path))
        assert path.check(dir=True)

    def test_subdirectory(self, tmpdir):
        path = tmpdir.join('foo').join('bar')
        make_directory(str(path))
        assert path.check(dir=True)

    def test_existing_directory(self, tmpdir):
        path = tmpdir.mkdir('foo')
        make_directory(str(path))

    def test_existing_file(self, tmpdir):
        path = tmpdir.join('foo.bar')
        path.ensure(file=True)
        with pytest.raises(OSError):
            make_directory(str(path))
