import pytest

from scanomatic.util.generic_name import get_generic_name


@pytest.mark.parametrize(
    "seed,expected",
    [
        (42, "Generic Grebe"),
        (1764, "Generic Guineafowl"),
    ]
)
def test_generic_name_from_int(seed, expected):
    assert get_generic_name(seed) == expected
