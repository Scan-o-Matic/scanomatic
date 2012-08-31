#!/usr/bin/env python

import unittest
import numpy as np
import sys
import inspect
from scipy.ndimage import convolve

import analysis_grid_cell_dissection as gc
import analysis_grid_array_dissection as ga
import analysis_grid_array as g

def simulate_colony(colony_thickness=30, i_shape=(105, 104), add_bg=True):

    i_shape = map(int, i_shape)

    origo = [i / 2 for i in i_shape]
    r = int((origo[0] * origo[1]) ** 0.5 / np.pi)

    i = gc.Blob(None, None, np.zeros(i_shape))

    i.set_blob_from_shape(circle=(origo, r))


    fgIM = np.random.normal(loc=colony_thickness, size=i_shape)

    im = np.zeros(i_shape)

    im[np.where(i.filter_array > 0)] = fgIM[np.where(i.filter_array \
                                    > 0)]

    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]], dtype=np.float)

    kernel /= kernel.sum()

    im = convolve(im, kernel)

    true_blob = im > 0 

    if add_bg:

        im += np.random.normal(loc=1, size=i_shape)
    
    return im, true_blob
    

def simulate_plate(pinning=[32, 48], colony_thickness=30, im_shape=None,
            grid_cell_size=None, simulate_8_bit=False, inverse_8_bit=False):

    if im_shape is None:

        if grid_cell_size is None:

            im_shape = [322*2, 482*2]
            grid_cell_size = [i / float(pinning[n]+2) for n, i in
                                enumerate(im_shape)]

        else:

            im_shape = [i * (pinning[d] + 2) for d, i in
                            enumerate(grid_cell_size)]
    """        
    if (min(im_shape) == im_shape[0]) != (min(pinning) == pinning[0]):

        pinning.reverse()

    if pinning[0] < pinning[1]:

        pinning.reverse()
        im_shape.reverse()
    """

    im = np.random.normal(loc=1, size=im_shape)

    a = 0.025 * colony_thickness

    for x in xrange(pinning[0]):

        for y in xrange(pinning[1]):

            blob_im, true_blob = simulate_colony(
                            colony_thickness=colony_thickness + \
                            a * np.random.normal(),
                            i_shape=grid_cell_size,
                            add_bg=False)

            low_x = int((x + 0.6) * grid_cell_size[0])
            low_y = int((y + 0.6) * grid_cell_size[1])

            im[low_x: low_x + int(grid_cell_size[0]),
                            low_y: low_y + int(grid_cell_size[1])] += \
                            blob_im

    im += np.random.normal(loc=1, size=im_shape)

    if simulate_8_bit:

        if im.min() < 0:
            im -= im.min()

        if im.max() > 255:

            im *= 255.0 / im.max()

        im = np.round(im).astype(np.int)

        if inverse_8_bit:

            im *= -1
            im += 255

    return im, grid_cell_size


class Test_Grid(unittest.TestCase):

    def setUp(self):

        self.pinning = [32, 48]
        self.grid_cell_size = [52, 105]

        self.colony_thickness = 500

        self.im, self.cell_size = simulate_plate(pinning=self.pinning,
                        colony_thickness=self.colony_thickness,
                        grid_cell_size=self.grid_cell_size,
                        simulate_8_bit=True, inverse_8_bit=True)

        pm_max_pos = int(max(self.pinning) == self.pinning[1])
        im_max_pos = int(max(self.im.shape) == self.im.shape[1])

        self.im_axis_order = [int(pm_max_pos != im_max_pos)]
        self.im_axis_order.append(int(im_axis_order[0] == 0)) 

        self.ga = ga.Grid_Analysis(None, self.pinning)

        self.ga.set_dim_order(self.im_axis_order)

    def test_find_grid(self):

        res = self.ga.get_analysis(self.im.copy())

        self.assertIsNot(self.cell_size, None)
        self.assertIn(self.cell_size[0], self.grid_cell_size)
        self.assertIn(self.cell_size[1], self.grid_cell_size)

        self.assertIsNot(self.ga.best_fit_frequency, None)

        self.assertEquals(map(round, self.cell_size), 
                map(round, self.ga.best_fit_frequency))

        low_x = int(1.1 * self.grid_cell_size[0])
        low_y = int(1.1 * self.grid_cell_size[1])
        self.assertEqual(low_x, res[0][0])
        self.assertEqual(low_y, res[1][0])


    def test_plate(self):

        grid = g.Grid_Array(None, 1, self.pinning)

        self.assertEquals([len(grid._grid_cells),
                len(grid._grid_cells[0])], self.pinning)

        f = grid.get_analysis(self.im.copy())
        cell_count_list = list()

        self.assertIsNot(f, None)
        
        for c_pos in ((0, 0), (31, 0), (0, 47), (31, 47)):

            for d in (0, 1):

                blob_center = f[c_pos[0]][c_pos[1]]['blob']['centroid']
                cell_count_list.append(f[c_pos[0]][c_pos[1]]['blob']['pixelsum'])

                grid_cell_size = grid._grid_cell_size

                self.assert_(grid_cell_size[d] * 0.25 < blob_center[d] < \
                        grid_cell_size[d] * 0.75, 
                        msg="Cell {0} has blob not in center".format(c_pos))

        cell_count = np.array(cell_count_list)
        cell_count /= cell_count.mean()
        cell_count = np.abs(1 - cell_count)

        self.assertEqual((cell_count < 0.0001).all(True), True)

class Test_Grid_Cell_Item(unittest.TestCase):

    Item = gc.Cell_Item

    def setUp(self):

        self.I = np.random.random((105, 104))
        self.Id = ["test", 1, [1, 3]]
        self.i = self.Item(None, self.Id, self.I)


    def test_load_no_parent(self):

        self.assertEqual((self.I == self.i.grid_array).sum(), 
                            self.I.shape[0] * self.I.shape[1])

    def test_identity(self):

        self.assertEqual(self.Id, self.i._identifier)

    def test_filter_shape(self):

        self.assertTupleEqual(self.I.shape, self.i.filter_array.shape)

    def test_logging_behaviour(self):

        self.i.logger.info("Test")
        self.i.logger.debug("Test")
        self.i.logger.warning("Test")
        self.i.logger.error("Test")
        self.i.logger.critical("Test")

    def test_get_round_kernel(self):

        refs = []
        tests = []

        for x in xrange(10):

            r = np.random.randint(5, 15)

            c = gc.get_round_kernel(radius=r)

            self.assertAlmostEqual(abs(1 - np.pi * r ** 2 / c.sum()), 0.000,
                                                                    places=1)

    def test_do_analysis(self):

        ret = self.i.do_analysis()

        self.assertEqual(ret, None)

        self.assertDictEqual(self.i.features, dict())

    def test_set_data_source(self):

        I2 = np.random.random((105, 104))

        self.i.set_data_source(I2)

        self.assertIs(self.i.grid_array, I2)
        self.assertIsNot(self.i.grid_array, self.I)

        self.assertEqual((I2 == self.i.grid_array).sum(),
                            I2.shape[0] * I2.shape[1])

        self.assertTupleEqual(I2.shape, self.i.filter_array.shape)

        self.assertGreater(np.abs(self.I - I2).sum(), 0)


class Test_Grid_Cell_Cell(Test_Grid_Cell_Item):

    Item = gc.Cell

    def setUp(self):

        self.i_shape = (105, 104)
        self.I = np.random.random(self.i_shape)
        self.Id = ["test", 1, [1, 3]]
        self.i = self.Item(None, self.Id, self.I)

    def test_filter(self):

        c = self.Item(None, None, self.I)

        self.assertEqual(c.filter_array.sum(), self.I.shape[0]*self.I.shape[1])

    def test_do_analysis(self):

        i = self.Item(None, None, self.I)

        ret = i.do_analysis()

        self.assertEqual(ret, None)

        k = sorted(i.features.keys())

        self.assertListEqual(k, sorted(('area', 'pixelsum', 'mean',
                        'median', 'IQR', 'IQR_mean')))

        self.assertEquals(i.features['area'], self.I.shape[0] * self.I.shape[1])
        self.assertAlmostEqual(i.features['mean'], self.I.mean(), places=3)

        self.assertAlmostEqual(i.features['median'], np.median(self.I),
                                                            places=3)

        self.assertEquals(i.features['pixelsum'], self.I.sum())

    def test_type(self):

        i = self.Item(None, None, self.I)

        self.assertEqual(i.CELLITEM_TYPE, 3)

class Test_Grid_Cell_Blob(Test_Grid_Cell_Item):

    Item = gc.Blob

    def setUp(self):

        self.i_shape = (105, 104)
        self.I = np.random.random(self.i_shape)
        self.Id = ["test", 1, [1, 3]]
        self.i = self.Item(None, self.Id, self.I)

        self.im, self.true_blob = simulate_colony()

    def test_type(self):

        i = self.Item(None, None, self.I)

        self.assertEqual(i.CELLITEM_TYPE, 1)

    def test_do_analysis(self):

        i = self.Item(None, None, self.I)

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

        i = self.Item(None, None, self.I)

        rect = [[21, 24], [56, 71]]
        r_area = (rect[1][0] - rect[0][0], rect[1][1] - rect[0][1])
 
        i.set_blob_from_shape(rect=rect)

        self.assertTupleEqual(i.filter_array.shape, self.I.shape)

        i.do_analysis()

        self.assertEqual(i.features['area'], r_area[0] * r_area[1])

    def test_full_empty_filters(self):

        i = self.Item(None, None, self.I)

        i.filter_array = np.zeros(self.i_shape)

        i.do_analysis()

        self.assertEqual(i.features['area'], 0)
        self.assertEqual(i.features['pixelsum'], 0)

        i.filter_array = np.ones(self.i_shape)

        i.do_analysis()

        self.assertEqual(i.features['area'], self.i_shape[0] * self.i_shape[1])
        self.assertEqual(i.features['pixelsum'], self.I.sum())

    def test_thresholds(self):

        locA = 2
        locB = 30
        locC = 200

        I1 = np.random.normal(locA, size=self.i_shape)
        I2 = np.random.normal(locB, size=self.i_shape)

        I = np.random.randint(0, 2, self.i_shape)

        I[np.where(I == 1)] = I1[np.where(I == 1)]
        I[np.where(I == 0)] = I2[np.where(I == 0)]

        i = self.Item(None, None, I)


        for t in xrange(-5, 10):

            i.set_threshold(threshold=t)

            self.assertEqual(i.threshold, t)

        t2 = np.random.randint(5, 15)
        i.set_threshold(threshold=t2, relative=True)

        self.assertEqual(t2 + t, i.threshold)

        i.set_threshold(threshold=0)
        i.set_threshold()

        self.assertLess(i.threshold, locB)
        self.assertGreater(i.threshold, locA)

        I3 = np.random.normal(locC, size=self.i_shape)

        II = I.copy()

        II[np.where(II < i.threshold)] = I3[np.where(II < i.threshold)]

        i.set_threshold(im=II)

        self.assertLess(i.threshold, locC)
        self.assertGreater(i.threshold, locB)

    def test_cirularity_of_filter(self):

        i = self.Item(None, None, self.I)

        i.filter_array[:,:] = 0
        c_center = (50, 50)
        c_r = 20
        circle = gc.points_in_circle((c_center, c_r))

        for pos in circle:

            i.filter_array[pos] = 1

        c_o_m, radius = i.get_ideal_circle(i.filter_array)
        c_o_m_fraction = sum([abs(1 - x[0] / x[1]) for x in \
            zip(c_o_m, c_center)])

        self.assertAlmostEqual(c_o_m_fraction, 0.000, places=1)
        self.assertAlmostEqual(radius/float(c_r), 1.00, places=1)
        self.assertAlmostEqual(i.get_circularity(), 1.00, places=1) 

        rect = [[21, 24], [56, 71]]
        r_area = (rect[1][0] - rect[0][0], rect[1][1] - rect[0][1])

        i.set_blob_from_shape(rect=rect)

        self.assertGreater(i.get_circularity(), 2.0)

        i.filter_array *= 0

        self.assertEqual(i.get_circularity(), 1000)

    def test_detect_threshold(self):

        self.im, self.true_blob = simulate_colony(colony_thickness=100,
                                                add_bg=True)

        i = self.Item(None, None, self.im)

        i.threshold_detect(threshold=20)        

        #np.save("_debug_t1.npy", self.true_blob)
        #np.save("_debug_t2.npy", i.filter_array)

        self.assertEqual(np.abs(self.true_blob - 
                    i.filter_array).sum(), 0)
                   
        i.threshold_detect(threshold=20, color_logic="inv")

        self.assertEqual(np.abs((self.true_blob == 0) - i.filter_array).sum(), 0)

        self.assertEqual(i.grid_array[np.where(i.filter_array > 0)].sum(),
                    i.grid_array[np.where(self.true_blob == 0)].sum())

        i.threshold_detect()

        self.assertEqual(np.abs(self.true_blob - i.filter_array).sum(), 0)

        self.assertEqual(i.grid_array[np.where(i.filter_array)].sum(),
                    i.grid_array[np.where(self.true_blob)].sum())

    def test_manual_detect(self):

        i = self.Item(None, None, self.I.copy())

        i.set_blob_from_shape(circle=((50, 50), 20))

        i_filter = i.filter_array.copy()

        self.assertIsNot(i_filter, i.filter_array)

        I2 = np.random.normal(40, size=self.i_shape)

        i.grid_array[np.where(i.filter_array)] = I2[np.where(i.filter_array)]

        i.manual_detect((45, 50), 25)

        self.assertGreater(np.abs(i.filter_array - i_filter).sum(), 0)
 
        i_filter = i.filter_array.copy()

        i.set_blob_from_shape(circle=((45, 50), 25))

        self.assertLess(np.abs(i.filter_array - \
                    i_filter).sum()/i_filter.sum(), 0.01)


    def test_detect(self):

        i = gc.Blob(None, None, self.I)
        tollerance = 100
        colony_thickness = 300
        #x = 1

        while colony_thickness > 3:

            self.im, self.true_blob = simulate_colony(colony_thickness)

            i.set_data_source(self.im)

            i.detect()

            diff = i.filter_array.astype(np.int) - \
                        self.true_blob.astype(np.int)

            #np.save("_debug_{0}.npy".format(x), diff)
            #x += 1
            
            diff = np.abs(diff).sum()

            self.assert_(diff < tollerance,
                msg="Failed at {0} thickness, {1} is more than {2}".format(
                colony_thickness, diff, tollerance))

            colony_thickness /= 2.0

class Test_Grid_Cell_Background(Test_Grid_Cell_Item):

    Item = gc.Background

    def setUp(self):

        self.i_shape = (105, 104)
        self.I = np.random.random(self.i_shape)
        self.blob = gc.Blob(None, None, self.I)
        self.Id = ["test", 1, [1, 3]]
        self.i = self.Item(None, self.Id, self.I, self.blob,
                        run_detect=False)

    def test_type(self):

        self.assertEqual(self.i.CELLITEM_TYPE, 2)

    def test_do_analysis(self):

        ret = self.i.do_analysis()

        self.assertEqual(ret, None)

        k = sorted(self.i.features.keys())

        self.assertListEqual(k, sorted(('area', 'pixelsum', 'mean',
                        'median', 'IQR', 'IQR_mean')))

        #self.assertEquals(i.features['area'], I.shape[0] * I.shape[1])
        #self.assertAlmostEqual(i.features['mean'], I.mean(), places=3)
        #self.assertAlmostEqual(i.features['median'], np.median(I), places=3)
        #self.assertEquals(i.features['pixelsum'], I.sum())

    def test_detect(self):

        r = 30
        self.blob.set_blob_from_shape(circle=((50, 50), r))

        self.i.detect()

        diff_filter = np.abs(self.blob.filter_array - 1)

        self.assertLessEqual(self.blob.filter_array.sum(),
                        diff_filter.sum())

        self.assertGreaterEqual(self.i.filter_array.shape[0] *
                    self.i.filter_array.shape[1] - r ** 2 * np.pi,
                    self.i.filter_array.sum())


class Test_Analysis_Recipe(unittest.TestCase):
    """NEED MORE TESTS"""

    def setUp(self):

        self.I = np.random.random((104, 105))
        self.b = gc.Blob(None, None, self.I)
        self.ar1 = gc.rblob.Analysis_Recipe_Empty(self.b)
        self.ar1.set_reference_image(self.I)

        self.ar2 = gc.rblob.Analysis_Recipe_Empty(self.b, parent=self.ar1)
        #self.ar3 = gc.rblob.Analysis_Recipe_Median_Filter(self.b, self.ar2)
        #self.ar4 = gc.rblob.Analysis_Recipe_Gauss_2(self.b, self.ar1)
        #self.ar5 = gc.rblob.Analysis_Recipe_Erode(self.b, self.ar1)

    def test_set_image(self):

        im = np.random.random((104, 105))
        id1 = id(self.ar1._analysis_image)

        self.ar1.set_reference_image(im, inplace=True)
        self.assertNotEqual(id(self.b.grid_array), id(self.ar1._analysis_image))
        self.assertNotEqual(id(self.I), id(self.ar1._analysis_image))
        self.assertEqual(id1, id(self.ar1._analysis_image))

        self.ar1.set_reference_image(im, inplace=False)
        self.assertNotEqual(id(im), id(self.ar1._analysis_image))
        self.assertNotEqual(id1, id(self.ar1._analysis_image))

        im2 = np.random.random((104, 105))
        id2 = id(im)

        self.ar2.set_reference_image(im2, inplace=True)
        self.assertNotEqual(id2, id(self.ar1._analysis_image))
        self.assertNotEqual(id2, id(self.ar2._analysis_image))
        self.assertEqual(np.abs(im2 - self.ar1._analysis_image).sum(), 0)
        self.assertEqual(np.abs(im2 - self.ar2._analysis_image).sum(), 0)

    def test_sub_analysis(self):

        pass


class Test_Derived_Analysis(unittest.TestCase):
    """NEED MORE TESTS"""

    def setUp(self):

        self.known_classes = [gc.__getattribute__(c) for c in dir(gc) if 
                    inspect.isclass(gc.__getattribute__(c))
                    and issubclass(gc.__getattribute__(c), 
                    gc.rblob.Analysis_Recipe_Abstraction)]

        self.I = np.random.random((104, 105))
        self.b = gc.Blob(None, None, self.I)

        self.guess_count = sum([1 for i in dir(gc) 
                if 'Analysis_' == i[:9]])
       
    def test_good_names(self):

        self.assertEqual(self.guess_count, len(self.known_classes))

    def test_init(self):

        for c in self.known_classes:

            ar1 = c(self.b, None)

            self.assertIs(ar1.parent, None)
            self.assertIs(ar1.grid_cell, self.b)
        
            ar1._do(self.I)


if __name__ == "__main__":

    unittest.main()
