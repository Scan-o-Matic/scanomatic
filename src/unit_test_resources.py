#!/usr/bin/env python

import unittest
import numpy as np
import sys
import inspect
import types

import resource_image as r_i

class Test_Image(unittest.TestCase):

    im_path = './unittests/test_img.tiff'
    im_dpi = 600
    test_target = './unittests/test_target.tiff'

    def setUp(self):

        pass

    def test_find_grid(self):

        im = r_i.Quick_Scale_To_im(self.im_path, source_dpi=self.im_dpi,
                target_dpi=150)

        self.assertNotEqual(type(im), types.IntType)


if __name__ == "__main__":

    unittest.main()
