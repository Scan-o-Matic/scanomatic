#!/usr/bin/env python
"""Resource module containing classes for handling fixtures."""

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson", "Andreas Skyman"]
__license__ = "GPL v3.0"
__version__ = "0.0995"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


#
# DEPENDENCIES
#

import os, os.path, sys
import types
import itertools
import logging
import numpy as np

#
# SCANNOMATIC LIBRARIES
#

import resource_image as img_base
import resource_config as conf

class Fixture_Settings():

    def __init__(self, fixture_config_root, fixture="fixture_a", image=None, marking=None, markings=-1):

        self.fixture_name = fixture
        self._fixture_config_root = fixture_config_root
        self.conf_location = self._fixture_config_root + os.sep + self.fixture_name + ".config"

        self.fixture_config_file = conf.Config_File(self.conf_location)


        if image != None:
            self.image_path = image
        else:
            self.image_path = None

        self.conf_location = self._fixture_config_root + os.sep + "current_image.tmp_config"
        self.current_analysis_image_config = conf.Config_File(self.conf_location)

        if marking != None:
            self.marking_path = marking
        else:
            self.marking_path = self.fixture_config_file.get("marker_path")

        if markings > -1:
            self.markings = markings
        else:
            self.markings = self.fixture_config_file.get("marker_count")
        self.mark_X = None
        self.mark_Y = None
        self.A = None


    def get_markers_from_conf(self):

        X = []
        Y = []

        if self.markings == 0 or self.markings is None:
            return None, None

        for m in xrange(self.markings):
            Z = self.get_setting('marking_{0}'.format(m))
            if Z is not None:
                X.append(Z[0])
                Y.append(Z[1])

        if len(X) == 0:
            return None, None
        
        return np.array(X), np.array(Y)

    def get_setting(self, key):

        return self.fixture_config_file.get(key)

    def marker_analysis(self, fixture_setup=False, output_function=None):

        if self.marking_path == None or self.markings < 1:
        
            msg = "Error, no marker set ('%s') or no markings (%s)." % (
                self.marking_path, self.markings)
            if output_function != None:
                output_function('Fixture calibration',msg,110, debug_level='error')
            else:
                logging.error(msg)

            return None

        if fixture_setup:
            analysis_img = self._fixture_config_root + os.sep + self.fixture_name + ".tiff"
            target_conf_file = self.fixture_config_file
        else:
            analysis_img = os.getcwd() + os.sep + "tmp_analysis_img.tiff"
            target_conf_file = self.current_analysis_image_config

        if self.image_path:
            img_base.Quick_Scale_To(self.image_path, analysis_img)
            self.image_path = None  #Hackish but makes sure scaling is done once


        self.A = img_base.Image_Analysis(path = analysis_img, pattern_image_path = self.marking_path)

        msg = "Finding pattern"
        if output_function != None:
            output_function('Fixture calibration',msg,10, debug_level='debug')
        else:
            logging.info(msg)

        Xs, Ys = self.A.find_pattern(markings = self.markings)

        self.mark_X = Xs
        self.mark_Y = Ys
       
        if len(Xs) == self.markings:
            for i in xrange(len(Xs)):
                target_conf_file.set("marking_" + str(i), (Xs[i], Ys[i]))
            target_conf_file.set("marking_center_of_mass", (Xs.mean(), Ys.mean())) 

        msg = "Showing image '{0}'".format(analysis_img)
        if output_function is None:
            logging.info(msg)
        else:
            output_function('Fixture calibration', msg, 10, debug_level='debug')

        return analysis_img

    def get_fixture_markings(self):
        i = 0
        m = True
        tmpX = []
        tmpY = []
        while m != None:
            m = self.fixture_config_file.get("marking_" + str(i))
            if m != None:
                tmpX.append(m[0])
                tmpY.append(m[1])
            i += 1
        return tmpX, tmpY

    def get_markings_rotations(self, X, Y):

        Mcom = (X.mean(), Y.mean())
        dX = X - Mcom[0]
        dY = Y - Mcom[1]

        L = np.sqrt(dX**2 + dY**2)

        tmpX, tmpY = self.get_fixture_markings()


        ref_X = np.array(tmpX)
        ref_Y = np.array(tmpY)

        ref_Mcom = self.fixture_config_file.get("marking_center_of_mass")
        ref_dX = ref_X - ref_Mcom[0]
        ref_dY = ref_Y - ref_Mcom[1]

        ref_L = np.sqrt(ref_dX**2 + ref_dY**2)

        if len(tmpY) == len(ref_X) == len(ref_Y):
            #Find min diff order
            s_reseed = range(len(ref_L))
            s = range(len(L))

            tmp_dL = []
            tmp_s = []
            for i in itertools.permutations(s):
                tmp_dL.append((L[list(i)]-ref_L)**2)
                tmp_s.append(i)

            dLs = np.array(tmp_dL).sum(1)
            s = list(tmp_s[dLs.argmin()])

            print "** Found sort order that matches the reference", s,". Error:", np.sqrt(dLs.min())

            #Quality control of all the markers so that none is bad
            #Later

            #Rotations
            A = np.arccos(dX/L)
            A = A * (dY > 0) + -1 * A * (dY < 0)

            ref_A = np.arccos(ref_dX/ref_L)
            ref_A = ref_A * (ref_dY > 0) + -1 * ref_A * (ref_dY < 0)

            dA = A[s] - ref_A

            d_alpha = dA.mean()
            print "** Found average rotation", d_alpha,"from set of delta_rotations:", dA

            return d_alpha, Mcom
        else:
            print "*** ERROR: Missmatch in number of markings"
            return None 

    def get_rotated_point(self, point, alpha, offset=(0,0)):
        tmp_l = np.sqrt(point[0] **2 + point[1]**2)
        tmp_alpha = np.arccos(point[0]/tmp_l)
        tmp_alpha = tmp_alpha * (point[1] > 0) + -1 * tmp_alpha * (point[1] < 0)
        new_alpha = tmp_alpha + alpha
        new_y = np.cos(new_alpha) * tmp_l + offset[0]
        new_x = np.sin(new_alpha) * tmp_l + offset[1]

        return (new_x, new_y)

    def set_areas_positions(self, X=None, Y=None):

        if X == None:
            X = self.mark_X
        if Y == None:
            Y = self.mark_Y
        
        alpha, Mcom = self.get_markings_rotations(X,Y)
        ref_Mcom = self.fixture_config_file.get("marking_center_of_mass")
 
        self.current_analysis_image_config.flush()

        ref_gs = self.fixture_config_file.get("grayscale_area")
        if ref_gs != None and bool(self.fixture_config_file.get("grayscale")) == True :
            dGs1 = np.array(ref_gs[0]) - ref_Mcom
            dGs2 = np.array(ref_gs[1]) - ref_Mcom

            self.current_analysis_image_config.set("grayscale_area", 
                [self.get_rotated_point(dGs1, alpha, offset=Mcom),
                self.get_rotated_point(dGs2, alpha, offset=Mcom)])

        i = 0
        ref_m = True
        while ref_m != None:
            ref_m = self.fixture_config_file.get("plate_" + str(i) +"_area")
            if ref_m != None:
                dM1 = np.array(ref_m[0]) - ref_Mcom
                dM2 = np.array(ref_m[1]) - ref_Mcom

                self.current_analysis_image_config.set("plate_" + str(i) + "_area", 
                    [self.get_rotated_point(dM1, alpha, offset=Mcom),
                    self.get_rotated_point(dM2, alpha, offset=Mcom)])

            i += 1

    def get_plates_list(self):

        plate_list = []

        p = True
        ps = "plate_{0}_area"
        i = 0

        while p is not None:

            p = self.fixture_config_file.get(ps.format(i))
            if p is not None:
                plate_list.append(i)

            i+= 1

        return plate_list
       
    def get_pinning_formats(self, plate):

        pt = "plate_{0}_pin_formats"
    
        return self.fixture_config_file.get(pt.format(plate))

    def set_pinning_formats(self, plate, pin_format):

        pt = "plate_{0}_pin_formats"
       
        t = self.get_pinning_formats(plate)
        if t is None:
            t = []

        if plate not in t:
            t.append(t)
            self.fixture_config_file.set(pt.format(plate), t) 
 
    def get_pinning_history(self, plate, pin_format):

        ph = "plate_{0}_pinning_{1}"

        return self.fixture_config_file.get(ph.format(plate, pin_format))

    def set_append_pinning_position(self, plate, pin_format, position):

        ph = "plate_{0}_pinning_{1}"
        h = self.get_pinning_history(plate, pin_format)
        if h is None:
            h = []
        h.append(position)

        self.fixture_config_file.set(ph.format(plate, pin_format), h)
        self.fixture_config_file.save()

    def set_pinning_positions(self, plate, pin_format, position_list):

        ph = "plate_{0}_pinning_{1}"
        self.fixture_config_file.set(ph.format(plate, pin_format), position_list)
        self.fixture_config_file.save()

    def reset_pinning_history(self, plate):

    
        ph = "plate_{0}_pinning_{1}"
        pts = self.get_pinning_formats(plate)
        if pts is not None:
            
            for t in pts:
                self.fixture_config_file.set(ph.format(plate, pin_format), [])
        
    def reset_all_pinning_histories(self):

        for p in self.get_plates_list():
            self.reset_pinning_history(p)

