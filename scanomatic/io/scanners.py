_SCANNERS = {
    'Test': {
        'name': 'Test',
        'power': True,
        'owner': None,
    },
}


def has_scanner(name):
    return name in _SCANNERS


def get(name):
    return _SCANNERS[name]


def get_free():
    return [
        scanner for scanner in _SCANNERS.values() if scanner['owner'] is None
    ]


def get_all():
    return _SCANNERS.values()
