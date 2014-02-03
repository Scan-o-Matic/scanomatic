#!/usr/bin/env python

import analysis_grid_cell_dissection as gc
import unit_test_analysis as uta

from time import time

im, blob = uta.simulate_colony()
i = gc.Blob(None, None, im)

st_time = time()

for x in xrange(1000):

    i.set_data_source(im)
    i.detect()

print "1000 Blobs in", time() - st_time

