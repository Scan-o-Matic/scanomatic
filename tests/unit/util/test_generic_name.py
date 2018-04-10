from __future__ import absolute_import

from scanomatic.util.generic_name import (
    get_generic_name, get_bird_list, get_adjective_list, get_adjective,
)


def test_get_generic_name():
    name = get_generic_name()
    name_list = get_bird_list()
    adjecive_list = get_adjective_list()
    assert name.split(' ', 1)[0].lower() in adjecive_list
    assert name.split(' ', 1)[1] in name_list


def test_get_name_list():
    name_list = get_bird_list()
    assert isinstance(name_list, list)
    assert len(name_list) > 1
    assert "Long-tailed tit" in name_list
    assert '' not in name_list


def test_get_adjective_list():
    adjectives = get_adjective_list()
    assert isinstance(adjectives, list)
    assert len(adjectives) > 1
    assert "polite" in adjectives
    assert '' not in adjectives


def test_get_adjective_alliterates():
    assert get_adjective('Test').startswith('t')


def test_get_adjective_always_gets_something():
    assert get_adjective('1')
