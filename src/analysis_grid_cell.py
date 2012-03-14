#!/usr/bin/env python
"""
Part of the analysis work-flow that holds the grid-cell object (A tile in a
grid-array with a potential blob at the center).
"""
__author__ = "Martin Zackrisson, Mats Kvarnstroem"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson", "Mats Kvarnstroem", "Andreas Skyman"]
__license__ = "GPL v3.0"
__version__ = "0.992"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


#
# DEPENDENCIES
#

import numpy as np
import math

#
# SCANNOMATIC LIBRARIES
#

import analysis_grid_cell_dissection as cell_dissection

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
    def __init__(self, identifier, center=(-1,-1), rectSize = (0,0), idtag = 'n/a', data_source=None):


        self._identifier = identifier
        self.center = np.asarray(center)
        self.rect = compute_rect(center,np.asarray(rectSize))
        self.idtag = idtag
        self.pinned = 1
        self.nr_neighbours = -1

        self.data_source = data_source

        self._previous_image = None

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

    def set_new_data_source_space(self, space='cell estimate',\
        bg_sub_source = None, polynomial_coeffs = None):

        if space == 'cell estimate':

            #DEBUG -> CELL ESTIMATE SPACE PART !
            #from matplotlib import pyplot as plt
            #plt.clf()
            #plt.imshow(self.data_source)
            #plt.title("Kodak Value Space")
            #plt.show() 
            #DEBUG END
            if bg_sub_source is not None:
                bg_sub = np.mean(self.data_source[np.where(bg_sub_source)])
                self.data_source = self.data_source - bg_sub
                self.data_source[np.where(self.data_source<0)] = 0
            #DEBUG -> CELL ESTIMATE SPACE PART !
            #from matplotlib import pyplot as plt
            #plt.clf()
            #plt.imshow(self.data_source)
            #plt.show() 
            #DEBUG END
            if polynomial_coeffs is not None:
                self.data_source = \
                    np.polyval(polynomial_coeffs, self.data_source)

            #DEBUG -> CELL ESTIMATE SPACE PART !
            #from matplotlib import pyplot as plt
            #plt.clf()
            #plt.imshow(self.data_source)
            #plt.title("Cell Estimate Space")
            #cb = plt.colorbar()
            #cb.set_label('Cells Estimate/pixel')           
            #plt.show() 
            #DEBUG END

        self.set_grid_array_pointers()
    
    def set_grid_array_pointers(self):
        for item_names in self._analysis_items.keys():
            self._analysis_items[item_names].grid_array = self.data_source

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

    def get_analysis(self, no_detect=False, no_analysis=False,
            remember_filter=False, use_fallback=False):
        """

            get_analysis iterates through all possible cell items
            and runs their detect and do_analysis if they are attached.

            The cell items' features dictionaries are put inside a 
            dictionary with the items' names as keys.

            If cell item is not attached, a None is put in the 
            dictionary to avoid key errors.

            Function takes one optional argument:

            @no_detect      If set to true, it will re-use the
                            previously used detection.

            @no_analysis    If set to true, there will be no
                            analysis done, just detection (if still
                            active). 

            @remember_filter    Makes the cell-item object remember the
                                detection filter array. Default = False.
                                Note: Only relevant when no_detect = False.

            @use_fallback       Optionally sets detection to iterative 
                                thresholding


        """
        
        features_dict = {}

        for item_name in self._analysis_item_names:
            if self._analysis_items[item_name]:
                self._analysis_items[item_name].\
                    set_data_source(self.data_source)
                if not no_detect:
                    self._analysis_items[item_name].detect(\
                        remember_filter = remember_filter, 
                        use_fallback_detection=use_fallback)
                if not no_analysis:
                    self._analysis_items[item_name].do_analysis()
                    features_dict[item_name] = \
                        self._analysis_items[item_name].features
            else:
                features_dict[item_name] = None


        return features_dict       

    def get_first_dim_as_tuple(self):
        return (self.rect[0], self.rect[0] + self.get_width())

    def get_second_dim_as_tuple(self):
        return (self.rect[1], self.rect[1] + self.get_height())

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

    def attach_analysis(self, blob=True, background=True, cell=True,\
        use_fallback_detection=False, run_detect=True, center=None, \
        radius=None):

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

            @center         A manually set blob centrum (if set
                            radius must be set as well)
                            (if not supplied, blob will be detected
                            automatically)

           @radius          A manually set blob radus (if set
                            center must be set as well)
                            (if not supplied, blob will be detected
                            automatically)
           

        """     

        if blob: 
            self._analysis_items['blob'] = cell_dissection.Blob(\
                [self._identifier , ['blob']], self.data_source, 
                use_fallback_detection=use_fallback_detection, 
                run_detect = run_detect, center=center, radius=radius)

        if background and self._analysis_items['blob']:
            self._analysis_items['background'] = cell_dissection.Background(\
                [self._identifier , ['background']], self.data_source, 
                self._analysis_items['blob'], run_detect=run_detect)

        if cell:
            self._analysis_items['cell'] = cell_dissection.Cell(\
                [self._identifier , ['cell']], self.data_source,
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

