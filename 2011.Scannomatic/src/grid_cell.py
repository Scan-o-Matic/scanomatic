#! /usr/bin/env python

# 
# colonies.py   v 0.1
#
# This is a convienience module for command line calling of all different types of colony
# analysis that are implemented.
#
# The module can also be imported directly into other scrips as a wrapper
#



#
# DEPENDENCIES
#

import numpy as np
import math

#
# SCANNOMATIC LIBRARIES
#

import grid_cell_analysis as gca

#
# Functions
#

def crop(A,rect):
    """ Crop numpy matrix/array A with rect = (x,y,width,height) where (x,y) is the topLeft corner 
        of the rect """
    return A[rect[1]:(rect[3]+rect[1]),rect[0]:(rect[2]+rect[0])]

def compute_rect(center,rectSize):
    """ Typically you call this function with rectSize = interDist which is the 
        default grid-rect size, but it might be the case that the size is set to smaller 
        manually by invoking method setRectSize in class GridArray.
        
        rectSize can also be a tuple (width,height)  
        
        for even, integer valued rectSize, the topLeft corner of the resulting rectangle will be 
        center-rectSize/2 , which results in a rect "biased upwards to the left"
        
        if rectSize is empty or has negative elements, an ndarray  with -1 elements is returned
    """
    center = np.asarray(center);
    #print rectSize
    if np.isscalar(rectSize):
        rectSize = np.asarray([rectSize,rectSize])
        #print "scalar rectsize"
    else:
        rectSize = np.asarray(rectSize)

    if (rectSize<0).any() or rectSize.size ==0:
        return np.asarray([-1, -1, -1, -1]);
    topLeft = center-rectSize/2.0
    #width = np.round(rectSize[0])-1;
    #height = np.round(rectSize[1])-1;
    return np.asarray(tuple(topLeft)+tuple(rectSize));

#
# SPREADSHEET COMPATIBILITY FUNCTIONS
#

def ystring(ycoor):
    """ 'ystring' produces a string representation for the ycoor (integer >=1)
        that corresponds to Excels column annotation.
        1 (first column) => 'A'
        2                => 'B'
        ...
        26               => 'Z'
        27               => 'AA'
        28               => 'AB' (etc
        
        Note that this is NOT the typical representation in another base/alphabet, 
        since in that case 'A' should correspond to 0, 'B' to 1, and 'Z' to 25, and
        'BA' to 26 since these would be short-hand to 'AAA' <=> 0, 'AAB' <=> 1, 'AAZ'<=> 25
        and 'ABA' to 26. Of course this could be offset:ed with +1, but it does not change 
        the fact that it is another algorithm for columns/rows >=27 
        
    """
    if ycoor<27:
        return chr(65 + ycoor-1)
    else:
        y = ycoor-1
        next = y/26
        rest = y%26
        return ystring(next)+chr(65+rest)

def create_id_tag(xcoor,ycoor,nrCols,nrRows):
    #x = xcoor-1;
    nrDigits = math.floor(math.log10(nrCols)+1)
    template = '%0'+('%d' % (nrDigits))+'d'
    return ystring(ycoor)+(template % (xcoor))


#
# CLASS: Grid_Cell
#

class Grid_Cell():
    def __init__(self, center=(-1,-1), rectSize = (0,0), idtag = 'n/a', data_source=None):

        self.center = np.asarray(center)
        self.rect = compute_rect(center,np.asarray(rectSize))
        self.idtag = idtag
        self.pinned = 1
        self.nr_neighbours = -1

        self.data_source = data_source

        self._analysis_items = {}
        self._analysis_item_names = ('blob','background','cell')
        for item_name in self._analysis_item_names:
            self._analysis_items[item_name] = None


    def __str__(self):

        return "%s id = %s centered at (x,y)=(%4.2f, %4.2f) with (width,height)=(%4.2f, %4.2f)" %\
            (self.__class__.__name__,self.idtag,self.center[0],self.center[1],self.rect[2],self.rect[3])
        #return "%s at center = (%d,%d) with size = (%d,%d)" % \
            #(self.__class__.__name__,self.center[0],self.center[1],self.rect[2],self.rect[3])     
    def __repr__(self):

        return self.__str__()

    #
    # SET functions
    #

    def set_data_source(self, data_source):

        self.data_source = data_source

    def set_rect_size(self, rect_size=None):

        if rect_size == None:
            rect_size = self.data_source.shape

        self.rect = compute_rect(self.center,rect_size)

    def set_center(self,center,rect_size = None):
        # Set the new center:
        self.center = np.asarray(center)
        # find the rectSize:
        if rect_size is None:
            rect_size = self.get_rect_size()

        # Set rectSize, which also corrects for the new center:
        self.set_rect_size(rect_size)

    def set_offset(self, offset_vec):

        vec = np.asarray(offset_vec)
        vec.shape = self.center.shape
        self.center += vec
        self.set_center(self.center)

    #
    # GET functions
    #

    def get_item(self, item_name):

        if item_name in self._analysis_items.keys():
            return self._analysis_items[item_name]
        else:
            return None

    def get_analysis(self):
        """

            get_analysis iterates through all possible cell items
            and runs their detect and do_analysis if they are attached.

            The cell items' features dictionaries are put inside a 
            dictionary with the items' names as keys.

            If cell item is not attached, a None is put in the 
            dictionary to avoid key errors.

            Function takes no arguments.

        """
        
        features_dict = {}

        for item_name in self._analysis_item_names:
            if self._analysis_items[item_name]:
                self._analysis_items[item_name].set_data_source(self.data_source)
                self._analysis_items[item_name].detect()
                self._analysis_items[item_name].do_analysis()
                features_dict[item_name] = self._analysis_items[item_name].features
            else:
                features_dict[item_name] = None

        return features_dict       

    def get_rect_size(self):
        return self.rect[2:4]

    def get_rect(self):
        return self.rect

    def get_top_left(self):

        return self.rect[0:2]

    def get_bottom_right(self):
        widthHeight = np.maximum(self.rect[2:4]-1,0)
        return self.rect[0:2]+widthHeight

    def get_bottom_left(self):
        x = self.rect[0]
        height = max(self.rect[3]-1,0)
        y = self.rect[1]+ height
        return np.array([x,y])

    def get_top_right(self):
        width = np.max(self.rect[2]-1)
        x = self.rect[0]+width
        y = self.rect[1]
        return np.array([x,y])

    def get_width(self):
        return self.rect[2]

    def get_height(self):
        return self.rect[3]

    #
    # Other functions
    #

    def attach_analysis(self, blob=True, background=True, cell=True, use_fallback_detection=False, run_detect=True):
        """
            attach_analysis connects the analysis modules to the
            Grid_Cell instance.

            Function has three optional boolean arguments:

            @blob           Attaches blob item (default)

            @background     Attaches background item (default)
                            Only possible if blob is attached

            @cell           Attaches cell item (default)

            @use_fallback_detection         Causes simple thresholding instead
                            of more sophisticated detection (default False)

            @run_detect     Causes the initiation to run detection

        """     

        if blob: 
            self._analysis_items['blob'] = gca.Blob(self.data_source, 
                use_fallback_detection=use_fallback_detection, 
                run_detect = run_detect)

        if background and self._analysis_items['blob']:
            self._analysis_items['background'] = gca.Background(self.data_source, 
                self._analysis_items['blob'], run_detect=run_detect)

        if cell:
            self._analysis_items['cell'] = gca.Cell(self.data_source,
                run_detect = run_detect)

    def detach_analysis(self, blob=True, background=True, cell=True):
        """

            detach_analysis disconnects the analysis modules specified
            from the Grid_Cell instance.

            Function has three optional boolean arguments:

            @blob           Detaches blob analysis (default)
                            This also detaches background

            @background     Detaches background analysis (default)

            @cell           Detaches cell analysis (default)

        """

        if blob:
            self._analysis_items['blob'] = None
            self._analysis_items['background'] = None

        if background:
            self._analysis_items['background'] = None

        if cell:
            self._analysis_items['cell'] = None

