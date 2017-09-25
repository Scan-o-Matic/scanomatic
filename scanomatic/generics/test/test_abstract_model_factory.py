import pytest

from scanomatic.generics.abstract_model_factory import AbstractModelFactory


class TestValidators:

    @pytest.mark.parametrize("num,expected", [
        (-1, True),
        (0, True),
        (2.42, True),
        (1j, False),
        ('a', False),
        (None, False)])
    def test_is_number(self, num, expected):
        assert AbstractModelFactory._is_real_number(num) == expected

    @pytest.mark.parametrize("tup,expected", [
        (tuple(), True),
        ([], True),
        ("foo", False),
        (42, False),
        (None, False)])
    def test_is_tuple_or_list(self, tup, expected):
        assert AbstractModelFactory._is_tuple_or_list(tup) == expected
