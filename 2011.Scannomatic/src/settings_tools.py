#
# DEPENDENCIES
#

import os, os.path, sys
import types
import itertools

import numpy as np

#
# SCANNOMATIC LIBRARIES
#

import image_analysis_base as img_base
import simple_conf as conf

class Fixture_Settings():

    def __init__(self, fixture_config_root, fixture="fixture_a", image=None, marking=None, markings=-1):

        self.fixture_name = fixture
        self._fixture_config_root = fixture_config_root
        conf_location = self._fixture_config_root + os.sep + self.fixture_name + ".config"

        self.fixture_config_file = conf.Config_File(conf_location)


        if image != None:
            self.image_path = image


        conf_location = self._fixture_config_root + os.sep + "tmp_current_image.config"
        self.current_analysis_image_config = conf.Config_File(conf_location)

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

    def marker_analysis(self, fixture_setup=False, output_function=None):

        if self.marking_path == None or self.markings < 1:
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

        msg = "Loading image"
        if output_function != None:
            output_function(msg)
        else:
            print msg

        self.A = img_base.Image_Analysis(path = analysis_img, pattern_image_path = self.marking_path)

        msg = "Finding pattern"
        if output_function != None:
            output_function(msg)
        else:
            print msg

        Xs, Ys = self.A.find_pattern(markings = self.markings, output_function = output_function)

        self.mark_X = Xs
        self.mark_Y = Ys
       
        if len(Xs) == self.markings:
            for i in xrange(len(Xs)):
                target_conf_file.set("marking_" + str(i), (Xs[i], Ys[i]))
            target_conf_file.set("marking_center_of_mass", (Xs.mean(), Ys.mean())) 
        print "*** Showing image", analysis_img
        return analysis_img

    def get_markings_rotations(self, X, Y):

        Mcom = (X.mean(), Y.mean())
        dX = X - Mcom[0]
        dY = Y - Mcom[1]

        L = np.sqrt(dX**2 + dY**2)

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
#                print tmp_dL[-1], i

            dLs = np.array(tmp_dL)
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

        print "*** Completed rotation and offset calculations. Ready to use!"

        

