#!/usr/bin/env python

import unittest
import analysis_grid_cell_dissection as gc
import numpy as np


class Test_Grid_Cell_Item(unittest.TestCase):

    Item = gc.Cell_Item

    def test_load_no_parent(self):

        I = np.random.random((100, 100))
        i = self.Item(None, None, I)

        self.assertEqual((I == i.grid_array).sum(), I.shape[0] * I.shape[1])

    def test_identity(self):

        Id = ["test", 1, [1, 3]]

        i = self.Item(None, Id, np.zeros((100, 100)))

        self.assertEqual(Id, i._identifier)

    def test_filter_shape(self):

        I = np.random.random(np.random.randint(50, 150, 2))

        i = self.Item(None, None, I)

        self.assertTupleEqual(I.shape, i.filter_array.shape)

    def test_logging_behaviour(self):

        i = self.Item(None, None, np.zeros((100, 100)))

        i.logger.info("Test")
        i.logger.debug("Test")
        i.logger.warning("Test")
        i.logger.error("Test")
        i.logger.critical("Test")

    def test_get_round_kernel(self):

        refs = []
        tests = []

        i = self.Item(None, None, np.zeros((100, 100)))

        for x in xrange(10):

            r = np.random.randint(5, 15)

            c = i.get_round_kernel(radius=r)

            self.assertAlmostEqual(abs(1 - np.pi * r ** 2 / c.sum()), 0.000,
                                                                    places=1)

    def test_do_analysis(self):

        i = self.Item(None, None, np.zeros((100, 100)))

        ret = i.do_analysis()

        self.assertEqual(ret, None)

        self.assertDictEqual(i.features, dict())

    def test_set_data_source(self):

        I1 = np.random.random((100, 100))

        i = self.Item(None, None, I1)

        I2 = np.random.random((105, 104))

        i.set_data_source(I2)

        self.assertEqual((I2 == i.grid_array).sum(),
                            I2.shape[0] * I2.shape[1])

        self.assertTupleEqual(I2.shape, i.filter_array.shape)

class Test_Grid_Cell_Cell(Test_Grid_Cell_Item):

    Item = gc.Cell

    def test_filter(self):

        I = np.zeros((105,104))

        c = self.Item(None, None, I)

        self.assertEqual(c.filter_array.sum(), I.shape[0]*I.shape[1])

    def test_do_analysis(self):

        I = np.random.random((105,104))

        i = self.Item(None, None, I)

        ret = i.do_analysis()

        self.assertEqual(ret, None)

        k = sorted(i.features.keys())

        self.assertListEqual(k, sorted(('area', 'pixelsum', 'mean',
                        'median', 'IQR', 'IQR_mean')))

        self.assertEquals(i.features['area'], I.shape[0] * I.shape[1])
        self.assertAlmostEqual(i.features['mean'], I.mean(), places=3)
        self.assertAlmostEqual(i.features['median'], np.median(I), places=3)
        self.assertEquals(i.features['pixelsum'], I.sum())

class Test_Grid_Cell_Blob(Test_Grid_Cell_Item):

    Item = gc.Blob

    def test_do_analysis(self):

        I = np.random.random((105,104))

        i = self.Item(None, None, I)

        ret = i.do_analysis()

        self.assertEqual(ret, None)

        k = sorted(i.features.keys())

        self.assertListEqual(k, sorted(('area', 'pixelsum', 'mean',
                        'median', 'IQR', 'IQR_mean', 'perimeter',
                        'centroid')))

        #self.assertEquals(i.features['area'], I.shape[0] * I.shape[1])
        #self.assertAlmostEqual(i.features['mean'], I.mean(), places=3)
        #self.assertAlmostEqual(i.features['median'], np.median(I), places=3)
        #self.assertEquals(i.features['pixelsum'], I.sum())

    def test_set_blob_from_shape_circle(self):

        I = np.random.random((105,104))

        i = self.Item(None, None, I)

        r = np.random.randint(10, 20)

        i.set_blob_from_shape(circle=((50, 50), r))

        self.assertTupleEqual(i.filter_array.shape, I.shape)

        i.do_analysis()

        self.assertAlmostEqual(abs(1 - np.pi * r ** 2 / i.features['area']), 0.000,
                                                                places=1)

    def test_set_blob_from_shape_rect(self):

        I = np.random.random((105,104))

        i = self.Item(None, None, I)

        rect = [[21, 24], [56, 71]]
        r_area = (rect[1][0] - rect[0][0], rect[1][1] - rect[0][1])
 
        i.set_blob_from_shape(rect=rect)

        self.assertTupleEqual(i.filter_array.shape, I.shape)

        i.do_analysis()

        self.assertEqual(i.features['area'], r_area[0] * r_area[1])

if __name__ == "__main__":

    unittest.main()
