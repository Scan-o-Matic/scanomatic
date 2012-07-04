#!/usr/bin/env python
"""Resource module for handling the aquired images."""

__author__ = "Martin Zackrisson, Andreas Skyman"
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

from PIL import Image
import sys, os
import types
from scipy.signal import fftconvolve
import numpy as np
import logging
import matplotlib.mlab as ml
import matplotlib.image as plt_img
import matplotlib.pyplot as plt

#
# SCANNOMATIC DEPENDENCIES
#

import resource_signal as r_signal

#
# FUNCTIONS
#

def Quick_Scale_To(source_path, target_path, source_dpi=600, target_dpi=150):

    small_im = Quick_Scale_To_im(source_path, source_dpi=source_dpi, 
        target_dpi=target_dpi)

    try:
        small_im.save(target_path)
    except:
        logging.error("Could not save scaled down image")
        return -1

def Quick_Scale_To_im(source_path, source_dpi=600, target_dpi=150):

    try:
        im = Image.open(source_path)
    except:
        logging.error("Could not open source")
        return -1

    small_im = im.resize((im.size[0]*target_dpi/source_dpi,im.size[1]*target_dpi/source_dpi), Image.BILINEAR)

    return small_im

def Quick_Invert_To_Tiff(source_path, target_path):
    import PIL.ImageOps
    try:
        im = Image.open(source_path)
    except:
        logging.error("Could not open source")
        return -1

    inv_im = PIL.ImageOps.invert(im)

    try:
        inv_im.save(target_path)
    except:
        logging.error("Could not save inverted image at " + str(target_path))
        return -1

    return True

def Quick_Rotate(source_path, target_path):
    try:
        im = Image.open(source_path)
    except:
        logging.error("Could not open source")

        return -1

    rot_im = im.rotate(90)

    try:
        rot_im.save(target_path)
    except:
        logging.error("Could not save inverted image at " + str(target_path))
        return -1

    return True


class Image_Analysis():
    def __init__(self, path=None, image=None, pattern_image_path=None):

        self._img = None
        self._pattern_img = None
        self._load_error = None
        self._transformed = False
        self._conversion_factor = 1.0

        if pattern_image_path:
            try:
                pattern_img = plt_img.imread(pattern_image_path)
            except:
                logging.error("Could not open orientation guide image at " + str(pattern_image_path))
                self._load_error = True

            if self._load_error != True:
                if len(pattern_img.shape) > 2:
                    pattern_img = pattern_img[:,:,0]
                self._pattern_img = pattern_img
                #self._pattern_img = np.ones((pattern_img.shape[0]+8,pattern_img.shape[1]+8))
                #self._pattern_img[4:self._pattern_img.shape[0]-4,4:self._pattern_img.shape[1]-4] = pattern_img


        if path:
            try:
                self._img = plt_img.imread(path)
            except:
                logging.error("Could not open image at " + str(path))
                self._load_error = True 

            if self._load_error != True:           
                if len(self._img.shape) > 2:
                    self._img = self._img[:,:,0]

        if image:
            if self._img:
                logging.warning("An image was already loaded (via path), replaced by passed on image")
            self._img = image

        #if self._img != None:
        #    self.set_correct_rotation()

    def load_other_size(self, path, conversion_factor):

        self._conversion_factor = float(conversion_factor)

        try:
            self._img = plt_img.imread(path)
        except:
            logging.error("Could not open image at " + str(path))
            self._load_error = True 

        if self._load_error != True:           
            if len(self._img.shape) > 2:
                self._img = self._img[:,:,0]


    def get_hit_refined(self, hit, conv_img):

        #quarter_stencil = map(lambda x: x/8.0, stencil_size)
        #m_hit = conv_img[(max_coord[0] - quarter_stencil[0] or 0):\
            #(max_coord[0] + quarter_stencil[0] or conv_img.shape[0]),
            #(max_coord[1] - quarter_stencil[1] or 0):\
            #(max_coord[1] + quarter_stencil[1] or conv_img.shape[1])]

        #mg = np.mgrid[:m_hit.shape[0],:m_hit.shape[1]]
        #w_m_hit = m_hit*mg *m_hit.shape[0] * m_hit.shape[1] / float(m_hit.sum())
        #refinement = np.array((w_m_hit[0].mean(), w_m_hit[1].mean()))
        #m_hit_max = np.where(m_hit == m_hit.max())
        #print "HIT: {1}, REFINED HIT: {0}, VECTOR: {2}".format(
            #refinement, 
            #(m_hit_max[0][0], m_hit_max[1][0]), 
            #(refinement - np.array([m_hit_max[0][0], m_hit_max[1][0]])))

        #print "OLD POS", max_coord, " ",
        #max_coord = (refinement - np.array([m_hit_max[0][0], m_hit_max[1][0]])) + max_coord
        #print "NEW POS", max_coord, "\n"

        return hit


    def get_convolution(self, img, marker, threshold=130):

        t_img = (img > threshold).astype(np.int8) * 2  - 1

        if len(marker.shape) == 3:
            marker = marker[:,:,0]
        t_mrk = (marker > 0) * 2 - 1

        return fftconvolve(t_img, t_mrk, mode='same')

    def get_best_location(self, conv_img, stencil_size, refine_hit=True):

        max_coord = np.where(conv_img == conv_img.max())
        if len(max_coord[0]) == 0:
            return None, conv_img

        max_coord = np.array((max_coord[0][0], max_coord[1][0]))

        #Refining
        if refine_hit:
            max_coord = self.get_hit_refined(max_coord, conv_img)

        #Zeroing out hit
        half_stencil = map(lambda x: x/2.0, stencil_size)
        conv_img[(max_coord[0] - half_stencil[0] or 0):\
            (max_coord[0] + half_stencil[0] or conv_img.shape[0]),
            (max_coord[1] - half_stencil[1] or 0):\
            (max_coord[1] + half_stencil[1] or conv_img.shape[1])] = 0

        

        return max_coord, conv_img


    def get_best_locations(self, conv_img, stencil_size, n, refine_hit=True, threshold=130):

        m_locations = []
        c_img = conv_img.copy()

        while len(m_locations) < n:

            m_loc, c_img = self.get_best_location(c_img, stencil_size, refine_hit)

            m_locations.append(m_loc)

        return m_locations

    def find_pattern(self, markings=3, img_threshold=130):

        if self.get_loaded():
            c1 = self.get_convolution(self._img, self._pattern_img, threshold=img_threshold)
            m1 = np.array(self.get_best_locations(c1, self._pattern_img.shape, markings, 
                refine_hit=False))

            return m1[:,1], m1[:,0]

        else:
            return None

    def get_loaded(self):
        return (self._img != None) and (self._load_error != True)


        

    def get_subsection(self, section):
        if self.get_loaded() and section is not None:
            if section[0][1] > section[1][1]:
                upper = section[1][1]
                lower = section[0][1]
            else:
                upper = section[0][1]
                lower = section[1][1]
            if section[0][0] > section[1][0]:
                left = section[1][0]
                right = section[0][0]
            else:
                left = section[0][0]
                right = section[1][0]

            try:
                subsection = self._img[int(left*self._conversion_factor):int(right*self._conversion_factor),
                    int(upper*self._conversion_factor):int(lower*self._conversion_factor)]
            except:
                subsection = None

            return subsection

        return None

class Analyse_Grayscale():
    def __init__(self, type="Kodak", image=None, scale_factor=1.0, dpi=600):

        self.grayscale_type = "Kodak"
        self._grayscale_dict = {\
            'Kodak': {\
                'aims':[82,78,74,70,66,62,58,54,50,46,42,38,34,30,26,22,18,14,10,6,4,2,0],
                'width': 55,
                'sections': 24,
                'lower-than-half-width':350,
                'higher-than-half-width':150,
                'length': 28.57
                }\
            }

        self._grayscale_aims = self._grayscale_dict[self.grayscale_type]['aims']
        self._grayscale_sections = self._grayscale_dict[self.grayscale_type]['sections']
        self._grayscale_width = self._grayscale_dict[self.grayscale_type]['width']
        self._img = image

        #Variables from analysis
        self._grayscale_pos = None
        self._grayscale = None
        self._grayscale_X = None

        if image != None:

            self._grayscale_pos, self._grayscale = self.get_grayscale(\
                scale_factor=scale_factor, dpi=dpi)
            self._grayscale_X = self.get_grayscale_X(self._grayscale_pos)

   
    def get_kodak_values(self):

        return self._grayscale_dict[self.grayscale_type]['aims']
 
    def get_grayscale_X(self, grayscale_pos=None):

        if grayscale_pos == None:
            grayscale_pos = self._grayscale_pos

        if grayscale_pos == None:
            return None
           
        X = np.array(grayscale_pos)
        median_distance =  np.median(X[1:] - X[:-1])

        self._grayscale_X = [0]

        pos = 0
        for i in range(1,len(grayscale_pos)):
            pos += int(round((grayscale_pos[i] - grayscale_pos[i-1]) / median_distance))
            self._grayscale_X.append(pos)

        return self._grayscale_X

    def get_grayscale(self, image=None, scale_factor=1.0, dpi=600):

        if image != None:
            self._img = image

        if self._img is None or sum(self._img.shape) == 0:
            return None

        #DEBUG PLOT
        #plt.imshow(self._img)
        #plt.show()
        #DEBUG PLOT END
    
        if scale_factor == 1 and dpi != 600:
            scale_factor = 600.0/dpi

        logging.debug("GRAYSCALE ANALYSIS: Of images {0} at dpi={1} and scale_factor={2}".format(\
            self._img.shape, dpi, scale_factor))

        A = (self._img[self._img.shape[0]/2:,:] < (256*1/4)).astype(int)
        #B = (self._img[:self._img.shape[0]/2,:] > (256*2/4)).astype(int)
        #DEBUG PLOT
        #plt.subplot(121)
        #plt.imshow(A)
        #plt.subplot(122)
        #plt.imshow(B)
        #plt.show()
        #DEBUG PLOT END

        rect = [[0,0],[self._img.shape[0],self._img.shape[1]]]
        orth_diff = -1

        kern = np.asarray([-1,0,1])
        Aorth = A.mean(0)
        Aorth_edge = abs( fftconvolve(kern, Aorth, 'same'))
        Aorth_edge_threshold = 0.2 #Aorth_edge.max() * 0.70
        Aorth_signals = Aorth_edge > Aorth_edge_threshold
        Aorth_positions = np.where(Aorth_signals)

        if len(Aorth_positions[0]) > 1:            
            Aorth_pos_diff = abs(1 - ((Aorth_positions[0][1:] - \
                Aorth_positions[0][:-1]) / float(self._grayscale_width)))
            rect[0][1] = Aorth_positions[0][Aorth_pos_diff.argmin()]
            rect[1][1] = Aorth_positions[0][Aorth_pos_diff.argmin()+1]
            orth_diff = rect[1][1] - rect[0][1]

        if orth_diff == -1:
            #Orthagonal trim second try
            firstPass = True
            in_strip = False
            threshold = 0.15
            threshold2 = 0.3

            for i, orth in enumerate(A.mean(0)):
                if firstPass:
                    firstPass = False
                else:
                    if abs(old_orth - orth) > threshold:
                        if in_strip == False:
                            if orth > threshold2:
                                rect[0][1] = i
                                in_strip = True
                        else:
                            rect[1][1] = i
                            break
                old_orth = orth


            orth_diff = rect[1][1]-rect[0][1]

        #safty margin
        min_orths = 30 / scale_factor
        if orth_diff > min_orths:
            rect[0][1] += (orth_diff - min_orths) / 2
            rect[1][1] -= (orth_diff - min_orths) / 2


        self._mid_orth_strip = rect[0][1] + (rect[1][1] - rect[0][1]) / 2

        #Paralell trim - right
        box_need = self._grayscale_dict['Kodak']['lower-than-half-width'] / scale_factor
        i = (rect[1][0] - rect[0][0])/2.0

        #DEBUG PLOT
        #plt.imshow(self._img[rect[0][0]:rect[1][0],rect[0][1]:rect[1][1]])
        #plt.show()
        #END DEBUG PLOT

        strip_values = self._img[rect[0][0]:rect[1][0],rect[0][1]:rect[1][1]].mean(1)
        A2 = strip_values > 125



        A2_where = np.where(A2 == True)[0]
        if A2_where.size > 1:
            A2_rights = np.append(A2_where[0], A2_where[1:] - A2_where[:-1])
            
            edges = np.argsort(A2_rights)
            edge_pos = -1
            while A2_rights[edges[edge_pos]] > box_need:
                right_edge = edges[edge_pos]
                if A2_rights[right_edge] > box_need and A2_rights[:right_edge+1].sum() > i:
                    rect[1][0] = A2_rights[:right_edge+1].sum() + \
                        self._grayscale_dict['Kodak']['length']/2.0
                    if rect[0][0] < 0:
                        logging.warning('GRAYSCALE, Overshot upper bound setting. Compensating')
                        rect[0][0] = self._img.shape[0]


                    break
                else:
                    edge_pos -= 1
                    if abs(edge_pos) == edges.size:
                        break

        box_need = self._grayscale_dict['Kodak']['higher-than-half-width'] / scale_factor
        A2 = A2 == False
        A2_where = np.where(A2 == True)[0]
        if A2_where.size > 1:
            A2_lefts = np.append(A2_where[0], A2_where[1:] - A2_where[:-1])
            edges = np.argsort(A2_lefts)
            edge_pos = -1
            while A2_lefts[edges[edge_pos]] > box_need:
                left_edge = edges[edge_pos]
                if A2_lefts[left_edge] > box_need and A2_lefts[:left_edge+1].sum() < i:
                    rect[0][0] = A2_lefts[:left_edge].sum() - \
                        self._grayscale_dict['Kodak']['length']/2.0
                    if rect[0][0] < 0:
                        logging.warning('GRAYSCALE, Overshot lower bound setting. Compensating')
                        rect[0][0] = 0

                    break
                else:
                    edge_pos -= 1
                    if abs(edge_pos) == edges.size:
                        break
        
        ###DEBUG CUT SECTION
        #plt.clf()
        #plt.imshow(self._img[rect[0][0]:rect[1][0],rect[0][1]:rect[1][1]])
        #plt.show()
        ###DEBUG END





        strip_values = self._img[rect[0][0]:rect[1][0],rect[0][1]:rect[1][1]].mean(1)


        threshold = 1.2
        kernel = [-1, 1] #old [-1,2,-1]

        up_spikes = np.abs(np.convolve(strip_values,kernel,"same")) > threshold
        
        up_spikes = r_signal.get_center_of_spikes(up_spikes)

        best_spikes = r_signal.get_best_spikes(up_spikes, 
            self._grayscale_dict['Kodak']['length']/scale_factor, tollerance=0.05, 
            require_both_sides=False)

        frequency = r_signal.get_perfect_frequency2(best_spikes, 
            self._grayscale_dict['Kodak']['length']/scale_factor)

        offset = r_signal.get_best_offset(\
            self._grayscale_dict['Kodak']['sections'],
            best_spikes, frequency=frequency)

        signal =  r_signal.get_true_signal(self._img.shape[0], 
            self._grayscale_dict['Kodak']['sections'],
            up_spikes, frequency=frequency,
            offset=offset)

        if signal is None:
            logging.warning("GRAYSCALE, no signal detected for f={0} and offset={1}\
 in best_spikes={2} from spikes={3}".format(frequency, offset, best_spikes,
                up_spikes))

            return None, None

        safety_buffer = 0.2
        if signal[0] - frequency + frequency*safety_buffer < 0:
            logging.warning("GRAYSCALE, the signal got adjusted one interval"\
                " due to lower bound overshoot")
            signal += frequency
        if signal[-2] - frequency*safety_buffer > strip_values.size:
            logging.warning("GRAYSCALE, the signal got adjusted one interval"\
                " due to upper bound overshoot")
            signal -= frequency

        safety_coeff = 0.5
        gray_scale = []
        gray_scale_pos = []
        self.ortho_half_height = self._grayscale_dict['Kodak']['width'] /2.0*safety_coeff
        for pos in xrange(signal.size-1):
            gray_scale_pos.append(signal[pos] - 0.5*frequency + rect[0][0])
            gray_scale.append(\
                self._img[gray_scale_pos[-1]-0.5*frequency*safety_coeff:\
                    gray_scale_pos[-1]+0.5*frequency*safety_coeff,
                    self._mid_orth_strip-self.ortho_half_height:\
                    self._mid_orth_strip+self.ortho_half_height].mean())

        #from matplotlib import pyplot as plt
        #print offset, frequency
        #strip_values = self._img[:,rect[0][1]:rect[1][1]].mean(1)
        #plt.plot(strip_values)
        #plt.plot(np.arange(up_spikes.size)+rect[0][0],up_spikes*strip_values.max())
        #plt.plot(np.arange(best_spikes.size)+rect[0][0],best_spikes*strip_values.max()*0.5)
        #plt.plot(np.arange(self._grayscale_dict['Kodak']['sections']+2)*frequency+offset+rect[0][0],np.ones(self._grayscale_dict['Kodak']['sections']+2)*0.5*strip_values.max(), '*')
        #plt.plot(np.asarray(gray_scale_pos), gray_scale, 'o')
        #plt.plot((rect[0][0], rect[1][0]),np.ones(2)*0.75*strip_values.max(),'*')
        #plt.show()

        #DEBUG PLOT
        #plt.imshow(self._img.T, cmap=plt.cm.gray)
        #plt.plot(gray_scale_pos, np.ones(len(gray_scale_pos))*self._mid_orth_strip -\
             #self.ortho_half_height ,
            #'r-' )
        #plt.plot(gray_scale_pos, np.ones(len(gray_scale_pos))*self._mid_orth_strip +\
             #self.ortho_half_height,
            #'r-' )
        #plt.plot(gray_scale_pos, np.ones(len(gray_scale_pos))*self._mid_orth_strip ,
            #'ro' )
        #plt.plot(gray_scale_pos, gray_scale, 'b--')
        #plt.show()
        #END DEBUG PLOT

        self._gray_scale_pos = gray_scale_pos
        self._gray_scale = gray_scale

        return gray_scale_pos, gray_scale

    def get_old_grayscale(self, image=None, scale_factor=1.0, dpi=600):
    
        if image != None:
            self._img = image

        if self._img == None:
            return None

        A  = (self._img < (self._img.max()/4)).astype(int)

        rect = [[0,0],[self._img.shape[0],self._img.shape[1]]]
        orth_diff = -1

        kern = np.asarray([-1,0,1])
        Aorth = A.mean(0)
        Aorth_edge = abs( fftconvolve(kern, Aorth, 'same'))
        Aorth_edge_threshold = 0.2 #Aorth_edge.max() * 0.70
        Aorth_signals = Aorth_edge > Aorth_edge_threshold
        Aorth_positions = np.where(Aorth_signals)

        if len(Aorth_positions[0]) > 1:            
            Aorth_pos_diff = abs(1 - ((Aorth_positions[0][1:] - \
                Aorth_positions[0][:-1]) / float(self._grayscale_width)))
            rect[0][1] = Aorth_positions[0][Aorth_pos_diff.argmin()]
            rect[1][1] = Aorth_positions[0][Aorth_pos_diff.argmin()+1]
            orth_diff = rect[1][1] - rect[0][1]

            ### DEBUG CONVOLVE FINDING 
            #print Aorth_pos_diff
            #plt.plot(np.asarray([Aorth_positions[0][Aorth_pos_diff.argmin()], 
            #    Aorth_positions[0][Aorth_pos_diff.argmin()+1]]), np.ones(2)*0.75,
            #    lw=5)
        #plt.plot(np.arange(len(Aorth)),Aorth)
        #plt.plot(np.arange(len(Aorth)), Aorth_edge)
        #plt.plot(np.arange(len(Aorth)), Aorth_signals)
        #plt.show()        
        ###DEBUG END

        if orth_diff == -1:
            #Orthagonal trim second try
            firstPass = True
            in_strip = False
            threshold = 0.15
            threshold2 = 0.3

            for i, orth in enumerate(A.mean(0)):
                if firstPass:
                    firstPass = False
                else:
                    if abs(old_orth - orth) > threshold:
                        if in_strip == False:
                            if orth > threshold2:
                                rect[0][1] = i
                                in_strip = True
                        else:
                            rect[1][1] = i
                            break
                old_orth = orth


            orth_diff = rect[1][1]-rect[0][1]

        #safty margin
        min_orths = 30 / scale_factor
        if orth_diff > min_orths:
            rect[0][1] += (orth_diff - min_orths) / 2
            rect[1][1] -= (orth_diff - min_orths) / 2


        self._mid_orth_strip = (rect[0][1] + rect[1][1]) / 2.0

        ###DEBUG UNCUT SECTION
        #plt.clf()
        #plt.imshow(A)
        #plt.show()
        #plt.savefig('gs_test1.png')
        ###DEBUG END

        #Paralell trim
        i = (rect[1][0] - rect[0][0])/2
        box_needed = 120 / scale_factor
        boxsize = 0
        transition = 0
        A2 = (self._img[rect[0][0]:rect[1][0],rect[0][1]:rect[1][1]]< np.median(self._img)).astype(int).mean(1)

        ###DEBUG CUT SECTION
        #plt.clf()
        #plt.imshow(self._img[rect[0][0]:rect[1][0],rect[0][1]:rect[1][1]])
        #plt.show()
        ###DEBUG END
        
        while i>0:
            if A2[i] > 0.2:
                if boxsize > box_needed and A2[i] > 0.3:
                    break
                elif transition < 2 / scale_factor:
                    transition += 1
                else:
                    boxsize = 0
                    transition = 0
            else:
                boxsize += 1
            i -= 1
            #print i, boxsize, A2[i]


        rect[0][0] = i #Some margin?
        #if rect[0][0] < 0:
        #    rect[0][0] = 0

        #plt.clf()
        #plt.imshow(self._img[rect[0][0]:rect[1][0],rect[0][1]:rect[1][1]].T)
        #plt.savefig('gs_test2.png')

        #plt.clf()
        #plt.plot(A2)
        #plt.savefig('gs_test_2b.png')
        #print rect
        #np.savetxt('dev.txt', self._img[rect[0][0]:rect[1][0],rect[0][1]:rect[1][1]])
        
        #producing the gray_scale values
        #np.savetxt('grayscalearea.array', self._img[rect[0][0]:rect[1][0],rect[0][1]:rect[1][1]])
        strip_values = self._img[rect[0][0]:rect[1][0],rect[0][1]:rect[1][1]].mean(1)
        target_count = 23
        target_length = 29 / scale_factor
        threshold = 1.2
        previous_spike = 0
        low_length_tolerance = target_length - 5 / scale_factor
        high_length_tolerance = target_length + 5 / scale_factor
        kernel = [1,-1] #old [-1,2,-1]

        lengths = []
        lengthpos = []
        gray_scale = []
        gray_scale_pos = []
        up_spikes = abs(np.convolve(strip_values,kernel,"same")) > threshold
        #plt.clf()
        #plt.plot(up_spikes*30)
        #plt.savefig('gs_test3.png')

        for x in xrange(up_spikes.shape[0]):
            if up_spikes[x] == True:
                if (x - previous_spike) > low_length_tolerance:
                    lengths.append(x- previous_spike)
                    lengthpos.append(x) 
                    previous_spike = x
                elif len(lengths) == 0:
                    previous_spike = x

        if len(lengths) > 0:
            tmpA = np.array(lengths)
            tmpLength = np.median(tmpA)
            found_first = False
            #print "* mLength", tmpLength, "Array:", lengths
            skip_next_pos = False
            for pos in xrange(len(lengths)):
                if not skip_next_pos:
                    #print pos, lenthspos[pos], lengths[pos]            
                    if low_length_tolerance < lengths[pos] < high_length_tolerance:
                        found_first = True
                        gray_scale.append(strip_values[lengthpos[pos]-tmpLength*3/4:lengthpos[pos]-tmpLength*1/4].mean())
                        gray_scale_pos.append(lengthpos[pos]-tmpLength*1/2)

                    elif found_first and low_length_tolerance < lengths[pos]/2 < high_length_tolerance:
                        gray_scale_pos.append(lengthpos[pos]-tmpLength*3/2)
                        gray_scale.append(strip_values[lengthpos[pos]-tmpLength*7/4:lengthpos[pos]-tmpLength*5/4].mean())

                        gray_scale_pos.append(lengthpos[pos]-tmpLength*1/2)
                        gray_scale.append(strip_values[lengthpos[pos]-tmpLength*3/4:lengthpos[pos]-tmpLength*1/4].mean())


                    elif found_first and low_length_tolerance < lengths[pos]/3 < high_length_tolerance:
                        gray_scale.append(strip_values[lengthpos[pos]-tmpLength*11/4:lengthpos[pos]-tmpLength*9/4].mean())
                        gray_scale_pos.append(lengthpos[pos]-tmpLength*5/2)

                        gray_scale.append(strip_values[lengthpos[pos]-tmpLength*7/4:lengthpos[pos]-tmpLength*5/4].mean())
                        gray_scale_pos.append(lengthpos[pos]-tmpLength*3/2)

                        gray_scale.append(strip_values[lengthpos[pos]-tmpLength*3/4:lengthpos[pos]-tmpLength*1/4].mean())
                        gray_scale_pos.append(lengthpos[pos]-tmpLength*1/2)

                    elif found_first:
                        if pos+1 < len(lengths) and low_length_tolerance < (lengthpos[pos+1]-lengthpos[pos-1]) < high_length_tolerance:
                            gray_scale.append(strip_values[lengthpos[pos-1]+tmpLength*1/4:lengthpos[pos-1]+tmpLength*3/4].mean())
                            gray_scale_pos.append(lengthpos[pos-1]+tmpLength*1/2)
                            skip_next_pos = True
                        else:
                            logging.warning( "* Lost track after",pos,"sections... extrapolating")
                            for pp in xrange(target_count-len(gray_scale)):
                                gray_scale.append(strip_values[lengthpos[pos-1]+tmpLength*pp+tmpLength*1/4:lengthpos[pos-1]+tmpLength*pp+tmpLength*3/4].mean())
                                gray_scale_pos.append(lengthpos[pos-1]+tmpLength*pp+tmpLength*1/2)
                            break
                else:
                    skip_next_pos = False
        #plt.plot(np.array(gray_scale_pos),np.array(gray_scale),'w*')
        #print " done spikes!"
        if len(gray_scale) > target_count:
            gray_scale_pos =  gray_scale_pos[:target_count]
            gray_scale =  gray_scale[:target_count]

        #If something bad happens towards the end it fills up with zeros
        #It offsets the X-values to match actual positions along the excised strip
        for i, pos in enumerate(gray_scale):
            if np.isnan(pos):
                gray_scale[i] = 0.0

            gray_scale_pos[i] += rect[0][0]

        #plt.clf()
        #plt.plot(np.array(gray_scale_pos), np.array(gray_scale))
        #plt.savefig('gs_test4.png')

        #print gray_scale_pos
        #print gray_scale
        return gray_scale_pos, gray_scale


