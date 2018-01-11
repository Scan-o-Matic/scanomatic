class Scanners:
    def __init__(self):
        self._scanners = {
            'Test': {
                'name': 'Test',
                'power': False,
                'owner': None,
            },
        }

    def has_scanner(self, name):
        return name in self._scanners

    def get(self, name):
        return self._scanners[name]

    def get_free(self):
        return [
            scanner for scanner in self._scanners.values()
            if scanner['owner'] is None
        ]

    def get_all(self):
        return list(self._scanners.values())
