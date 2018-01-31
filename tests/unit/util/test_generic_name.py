from __future__ import absolute_import

from scanomatic.util.generic_name import get_generic_name, get_name_list


def test_get_generic_name():
    name = get_generic_name()
    name_list = get_name_list()
    assert name.split(' ', 1)[0] == "Generic"
    assert name.split(' ', 1)[1] in name_list


def test_get_name_list():
    name_list = get_name_list()
    assert isinstance(name_list, list)
    assert len(name_list) > 1
    assert "Long-tailed tit" in name_list
