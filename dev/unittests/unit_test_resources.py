#!/usr/bin/env python

import unittest
from matplotlib.pyplot import imread
import numpy as np
import sys
import inspect
import types

import resource_image as r_i

def uniq(seq):
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if x not in seen and not seen_add(x)]

class Test_Image(unittest.TestCase):

    im_path = './src/unittests/test_img.tiff'
    im_dpi = 600
    test_target = './src/unittests/test_target.tiff'
    pattern_image_path = './src/images/orientation_marker_150dpi.png'
    known_positions_600dpi = ((379, 140), (242, 5596), (2714, 3037))

    def setUp(self):

        self.im = r_i.Quick_Scale_To_im(self.im_path, source_dpi=self.im_dpi,
                target_dpi=150)

        self.ia = r_i.Image_Analysis(image=self.im,
                pattern_image_path=self.pattern_image_path)

    def test_load_image(self):

        self.assertEqual(self.ia.get_loaded(), True)

    def test_convolution(self):

        conv = self.ia.get_convolution()

        self.assertEquals(conv.shape, self.ia._img.shape)

    def test_find_best_hits(self):

        conv = self.ia.get_convolution()
        stencil_size = self.ia._pattern_img.shape

        locs = self.ia.get_best_locations(conv, stencil_size, 3, refine_hit=True)

        self.assertEqual(len(locs), 3)

        conv = self.ia.get_convolution()

        locs = self.ia.get_best_locations(conv, stencil_size, 3, refine_hit=False)

        self.assertEqual(len(locs), 3)

        
    def test_find_pattern(self):

        D1, D2 = self.ia.find_pattern()

        self.assertNotEqual(D1, None)
        self.assertNotEqual(D2, None)
        self.assertEqual(len(D1), 3)
        self.assertEqual(len(D2), 3)

        known_positions_150dpi = [[v/4.0 for v in p] for p in self.known_positions_600dpi]
        found_positions = zip(D1, D2)
        best_pos = [None] * 3
        used_pos = [None] * 3

        for i, pos in enumerate(found_positions):

            candidates = [max(abs(pos[0] - ref[0]), abs(pos[1] - ref[1]))
                for ref in known_positions_150dpi]

            best_pos[i] = min(candidates)
            for j in xrange(3):
                if candidates[j] == best_pos[i]:
                    used_pos[i] = j
                    break

        self.assertEqual(len(uniq(used_pos)), len(used_pos))

        ok_dist = [p <= 3 for p in best_pos]

        self.assertNotEqual(False in ok_dist, True)


class Test_Image_Grayscale(unittest.TestCase):
 
    im_path = './src/unittests/test_gs.tiff'
    im = imread(im_path)

    def setUp(self):
    
        self.ag = r_i.Analyse_Grayscale(target_type="Kodak", image=self.im,
                    scale_factor=1.0, dpi=600)

    def test_get_target_values(self):

        targets = self.ag.get_target_values()

        self.assertGreater(len(targets), 2)
        self.assertEqual(len(targets), self.ag._grayscale_sections)

    def test_get_grayscale_X(self):

        gs_X = self.ag.get_grayscale_x()

        self.assertNotEqual(gs_X, None)

        self.assertEqual(len(gs_X), len(self.ag.get_target_values()))

    def test_get_grayscale(self):

        gs_pos, gs_val = self.ag.detect_grayscale()

        self.assertEqual(len(gs_pos), len(gs_val))
        self.assertEqual(len(gs_pos), self.ag._grayscale_sections)

if __name__ == "__main__":

    unittest.main()
