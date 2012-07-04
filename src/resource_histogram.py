#!/usr/bin/env python
"""Resource module for analysis of images"""

__author__ = "Mats Kvarnstroem, Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Mats Kvarnstroem", "Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.0995"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import numpy as np
import types

#
# CLASSES
#

class Histogram():
    def __init__(self, img, run_at_init=True):


        self.labels = None
        self.counts = None
    
        if run_at_init:
            self.re_hist(img)

        #If image has more than one chanel it is trimmed to the first
        #It is possible to use img.mean(2).as_type(int) I think if it is preffered

    def re_hist(self, img):
        if type(img)==types.ListType:
            img = np.asarray(img)

        if len(img.shape) == 3:
            img = img[:,:,0]

        self.__checkSupport(img)

        self.nPixels = img.size
        self.labels, self.counts = self._hist(img)
        return (self.labels, self.counts)
    
    #def __repr__(self):
    #    return "(%d,%d)"%(self.counts,self.labels)   

    def __checkSupport(self, img):
        
        if len(img.shape)==3:
            
            raise NotImplemented("Support for color images is not yet supported by 'histogram' ")

    def counts(self):
        return self.counts
    
    def labels(self):
        return self.labels 
      
            
    def _hist(self,img):
        # if image is of type uint8:
        counts = np.histogram(img,bins=range(257))[0]
        labels = np.arange(256)
        # Notice the rather peculiar organization of the return arguments from np.histogram where 'bins'
        # has size len(counts)+1 size 'bins' is the bin EDGES
        return (labels, counts)
        
def otsu(histogram):
    """ threshold = otsu(labels,counts)
        
        Computes an optimal threshold according to Otsu's non-parametric method for two classes
        The output 'threshold' is the index to label should be interpreted as class0 = {labels<=threshold}
        and class1 = {lables>threshold}
    """
    labels, counts = histogram

    #labels, counts = np.float32(self.bins),np.float32(self.counts)
    labels, counts = np.float32(labels),np.float32(counts)
    # First compute muT = 'overall mean', mu2T= 'mean square' and sumc ='sum of counts'
    muT = mu2T = sumT = 0.0
    count = counts[0]
    label = labels[0]
    nSlots = len(labels)
    for k in range(nSlots):
        count = counts[k]
        label = labels[k]
        muT += count*label
        mu2T += count*label**2
        sumT += count

    muT = muT/sumT
    mu2T = mu2T/sumT
    #S2T = mu2T-muT**2
    
    critValue = -1.0
    threshold = -1
    # Start checking through the labels/bins:
    #     Index 0 means properties of the zero:th class and 
    #     index 1 for the second(upper) class:
    # The first check is for threshold = 0 so that only the first element is in the zero:th class
    w0 = 0
    w1 = sumT
    mu0 = 0  
    mu1 = muT 
    
    for k in range(nSlots):
        if counts[k]>0:
            wchange = counts[k]
            label = labels[k]
            change = wchange*label
            # incrementally change the values of mu0 and mu1 (first non-normalized):
            mu0 = (mu0*w0+change)
            mu1 = (mu1*w1-change)
            w0 = w0+wchange
            w1 = w1-wchange
            # normalize:
            mu0 = mu0/w0
            if w1>0:  # if w1==0, we are finished
                
                mu1 = mu1/w1
                critCand = w0*w1*(mu1-mu0)**2
                if critCand>critValue:
                    critValue= critCand
                    threshold = label
                    mu0_opt = mu0
                    mu1_opt = mu1
                    w0_opt = w0
                    w1_opt = w1
                    S2B_opt = critValue/(sumT**2)
                    
    return threshold#, mu0_opt, mu1_opt, w0_opt,w1_opt, S2B_opt
    
        
      
if __name__ == "__main__":
 
    import os
    from matplotlib import pyplot as pp
    #imgpath = 'D:\\Images\\P_COLONIES\\20111111_NaCl_WithBox'
    #imgfile = 'New_rig_NaCl_test_with_box_121.jpg'
    imgpath = 'D:\\Images\\P_VQM'
    imgfile = 'beach.jpg'
    img = np.flipud(pp.imread(os.path.join(imgpath,imgfile)))

    hist = histogram(img[:,:,1])
    counts,labels = hist.counts, hist.labels






