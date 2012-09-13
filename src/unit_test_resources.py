#!/usr/bin/env python

import unittest
import numpy as np
import sys
import inspect
import types

import resource_image as r_i

class Test_Image(unittest.TestCase):

    im_path = './src/unittests/test_img.tiff'
    im_dpi = 600
    test_target = './src/unittests/test_target.tiff'
    pattern_image_path = './src/images/orientation_marker_600dpi.png'


    def setUp(self):

        self.ia = r_i.Image_Analysis(path=self.im_path,
                pattern_image_path=self.pattern_image_path)

    def load_image(self):

        self.assertEqual(ia.get_loaded(), True)

    def test_scale(self):

        im = r_i.Quick_Scale_To_im(self.im_path, source_dpi=self.im_dpi,
                target_dpi=150)

        self.assertNotEqual(type(im), types.IntType)

    def test_convolution(self):

        conv = self.ia.get_convolution()

        self.assertEquals(conv.shape, ia._img.shape)

    def test_find_pattern(self):

        D1, D2 = self.ia.find_pattern()

        self.assertNotEqual(D1, None)
        self.assertNotEqual(D2, None)
        self.assertEqual(len(D1), 3)
        self.assertEqual(len(D2), 3)
        print D1, D2
if __name__ == "__main__":

    unittest.main()
