#!/usr/bin/env python
from importlib import import_module

som = import_module('scanomatic')
print 'Scan-o-Matic {0}, {1}'.format(som.get_version(), som.get_branch())