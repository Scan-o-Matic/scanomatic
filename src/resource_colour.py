import numpy as np
from matplotlib import pyplot as plt
import types
import logging
from time import time
import analysis_grid_cell_dissection as cell
import analysis_grid_array_dissection as grid
import resource_signal as r_signal

"""
#
# STUFF FROM SKETCHBOOK
#

im_path = "./2012-05-31_Iodine_LYS+_C.tif"
im_path2 = "./2012-05-31_Iodine_URA+_C.tif"

im_norm_low = 0
im_norm_high = 13

im = plt.imread(im_path)
im2 = plt.imread(im_path2)


#Debanding
im_deband = de_band_image(im, 0, 13)
im2_deband = de_band_image(im2, 0, 13)
#im2_lvld = level_image(im2_deband, im, [[0,13],[0,im2_deband.shape[1]]],
#    [[0,13],[0,im_deband.shape[1]]])
#Cropping away bad stuff
im_cropped = get_im_section(im_deband, margin=[[70,37],[50,60]])
im2_cropped = get_im_section(im2_deband, margin=[[70,42],[45,52]])

g = grid.Grid_Analysis(parent)
im_grid = g.get_analysis(get_gs_im(im_cropped), pinning_matrix, use_otsu = True,
        median_coeff=None, verboise=False, visual=False,
        history=[])

col_span = (im_grid[1].min(), im_grid[1].max())
row_span = (im_grid[0].min(), im_grid[0].max())

fig = plt.figure()
ax = fig.add_subplot(1,1,1, title="Testing gridding")
ax.imshow(im_cropped)
for column in im_grid[0]:
    line = plt.Line2D((column, column), col_span)
    ax.add_line(line)

for row in im_grid[1]:
    line = plt.Line2D(row_span, (row, row))
    ax.add_line(line)


#
# Pinned image
#
im_path3 = "./120613_Iodine_LYS+_5min_1.tif"
im3 = plt.imread(im_path3)
im3_deband = de_band_image(im3, 0, 200)
im3_cropped = get_im_section(im3_deband, margin=[[270,100],[300,240]])
pinning_matrix = (12, 8)

neg_rgb = []

def add_texts_to_fig(fig, features, f=None, f2=None, dist=10):

    ax = fig.gca()

    for k in features.keys():
        if f is not None:
            ax.text(k[1]+dist, k[0]-dist, str(f(features[k])), bbox=dict(facecolor='white', alpha=0.5))
        elif f2 is not None:
            ax.text(k[1]+dist, k[0]-dist, str(f2(im, k, features[k])), 
                bbox=dict(facecolor='white', alpha=0.5),
                fontsize='xx-small')
        else:
            ax.text(k[1], k[0], str(features[k]), bbox=dict(facecolor='white', alpha=0.5))

    return fig

#Calling the functions
neg_rgb = get_rgbs_from_rect_list(im3_cropped, neg_ctrl)
neg_vector = np.array(neg_rgb).mean(0)
pos_rgb = get_rgbs_from_rect_list(im3_cropped, pos_ctrl)
pos_vector = np.array(pos_rgb).mean(0)
ref_features = get_features_from_rects_and_rgbs(neg_ctrl, neg_rgb)
ref_features = get_features_from_rects_and_rgbs(pos_ctrl, pos_rgb, features = ref_features)

im_grid, grid_size = get_grid_for_im(im3_cropped, pinning_matrix)
features, fig = get_features_and_figure(im3_cropped, im_grid, pinning_matrix, grid_size)
ref_features2 = get_relative_feature_projections(im3_cropped, ref_features, neg_vector, pos_vector, features_are_rgb=True, length_normed=True)
features2 = get_relative_feature_projections(im3_cropped, features, neg_vector, pos_vector, length_normed=True)
fig = get_feature_plot(im3_cropped, features2, f=get_2_decimals, title="Brightness-Normed Phenotypes", dist=10)
fig = add_texts_to_fig(fig, ref_features2, f=get_2_decimals, dist=10)

#new code
pos_ctrl = [None, None]
neg_ctrl = [None, None]
pos_ctrl[0] = [[20,40],[600,635]]
pos_ctrl[1] = [[20,40],[705,740]]
neg_ctrl[0] = [[10,40],[395,425]]
neg_ctrl[1] = [[20,45],[490,535]]

cim = r_colour.Color_Image("./120613_Iodine_LYS+_5min_1.tif", parent=r_colour.Parent(level='info'))
cim.set_deband_image(0,200)
cim.set_crop_image(margin=[[270,100],[300,240]])

cim.set_features_from_rects(pos_ctrl, settings={'control': True, 'control-type': True})
cim.set_features_from_rects(neg_ctrl, settings={'control': True, 'control-type': False})
cim.set_grid(interactive=False)
#cim.set_grid()


cim.set_features_from_grid()
#cim.get_plt_features_detected().show()
cim.set_references_from_list(["5:2", "2:2", "3:0", "5:0"])
p = cim.get_phenotypes()
cim.get_plt_features(p, f=r_colour.get_2_decimals).show()

pos_ctrl = [None, None]
neg_ctrl = [None, None]
pos_ctrl[0] = [[30,50],[610,640]]
pos_ctrl[1] = [[25,45],[725,760]]
neg_ctrl[0] = [[20,55],[410,440]]
neg_ctrl[1] = [[25,55],[510,520]]
cim = r_colour.Color_Image("./120613_Iodine_LYS+_5min_2.tif", parent=r_colour.Parent(level='info'))
cim.set_deband_image(0,180)
cim.set_crop_image(margin=[[260,100],[330,200]])
"""

def get_2_decimals(f):

    return round(f,2)


class Parent():

    def __init__(self, level='warning'):

        levels = {'critical': logging.CRITICAL,
            'error': logging.ERROR,
            'warning': logging.WARNING,
            'info': logging.INFO,
            'debug': logging.DEBUG }

        self.logger = logging.getLogger('RESOURCE COLOUR')
        self.logger.setLevel(levels[level])

class Color_Image():

    def __init__(self, im_path, parent=None, external_fig=True, pinning_matrix = (12,8)):

        if parent is None:

            parent = Parent()

        self.logger = parent.logger
        self._parent = parent

        self._im_path = im_path
        self._im_original = plt.imread(im_path)
        self._im_debanded = None
        self._im_cropped = None
        self._im_leveled = None
        self._cur_im = "original"

        self._external_fig = external_fig

        self._filter_array = None
        self._filter_blobs = None
        self._filter_qualities = None
        self._filter_c_o_m = None

        self._pinning_matrix = pinning_matrix
        self._grid = None
        self._grid_cell_size = None
        self._features = []
        self._untrusted = []
        self._next_feature_identifier = 0

    def set_deband_image(self, im_norm_low, im_norm_high):
        self._im_debanded = self.get_cur_im(get_copy=True)
        norm_profile = self._im_debanded[im_norm_low:im_norm_high,:,:].mean(0)
        norm_median = np.median(norm_profile, 0)
        im_correction = norm_profile - norm_median
        for line in xrange(self._im_debanded.shape[0]):
            self._im_debanded[line,:,:] -= im_correction
        self._cur_im = "debanded" 

    def set_crop_image(self, rect=None, margin=None):
        self._im_cropped = self.get_im_section(rect=rect, margin=margin).copy()
        self._cur_im = "cropped"

    def set_level_image(ref_im, im_norm_rect, im_ref_norm_rect, gray=True):

        im = self.get_cur_im()
        im_profile = im[im_norm_rect[0][0]:im_norm_rect[0][1],
            im_norm_rect[1][0]:im_norm_rect[1][1]].mean(0).mean(0)
        im_ref_profile = ref_im[im_ref_norm_rect[0][0]:im_ref_norm_rect[0][1],
            im_ref_norm_rect[1][0]:im_ref_norm_rect[1][1]].mean(0).mean(0)

        if gray:
            self._im_leveled =  im - (im_ref_profile - im_profile).mean()
        else:
            rel_profile = im_ref_profile / im_profile
            self._im_leveled =  im * rel_profile
        self._cur_im = "leveled"

    def set_cur_im(self, im_name):

        if im_name == "original" and self._im_original is not None:
            self._cur_im = im_name
            return True
        elif im_name == "debanded" and self._im_debanded is not None:
            self._cur_im = im_name
            return True
        elif im_name == "leveled" and self._im_leveled is not None:
            self._cur_im = im_name
            return True
        elif im_name == "cropped" and self._im_cropped is not None:
            self._cur_im = im_name
            return True
        else:
            return False

    def get_fig(self, external_fig=None):

        if external_fig is None:
            external_fig = self._external_fig

        if external_fig:
            return plt.figure()
        else:
            return plt.Figure()

    def get_cur_im(self, get_copy=False):

        if self._cur_im == "original" and self._im_original is not None:
            im = self._im_original
        elif self._cur_im == "debanded" and self._im_debanded is not None:
            im = self._im_debanded
        elif self._cur_im == "leveled" and self._im_leveled is not None:
            im = self._im_leveled
        elif self._cur_im == "cropped" and self._im_cropped is not None:
            im = self._im_cropped
        else:
            return None

        if get_copy:
            return im.copy()
        else:
            return im

    def get_gs_im(self, rect=None, use_ref_vector=None):

        if rect is None:
            im = self.get_cur_im()
        else:
            im = self.get_im_section(rect=rect)    

        if use_ref_vector is None:
            return im.mean(2)
        else:

            if use_ref_vector.upper() == "NEG":
                n_vector = np.array([f.get_rgb_unit_vector() for f in self.get_neg_features()])
            elif use_ref_vector.upper() == "POS":
                n_vector = np.array([f.get_rgb_unit_vector() for f in self.get_pos_features()])
            elif use_ref_vector.upper() == "REF":
                n_vector = np.array([f.get_rgb_unit_vector() for f in self.get_ref_features()])

            if len(n_vector.shape) == 2:
                n_vector = n_vector.mean(0)

            return (im * n_vector).mean(2)


    def get_im_section(self, rect=None, margin=None):

        im = self.get_cur_im()
        if im is None:
            return None

        if rect:
            return im[rect[0][0]:rect[0][1], rect[1][0]:rect[1][1]]
        
        if margin:
            if type(margin) == types.IntType:
                return im[margin:im.shape[0]-margin, margin: im.shape[1]-margin]
            else:
                section = im[margin[0][0]:im.shape[0] - margin[0][1], 
                    margin[1][0]:im.shape[1] - margin[1][1]]
                return section

    def get_plts_grid(self, im_count, need_even=False):

        c = np.sqrt(im_count/6)

        y = int(2*c)
        x = int(3*c)

        if need_even:
            y += y%2
            x += x%2

        d = im_count - y*x
        while d > 0:
            if y < x and d > y:
                y += 1
                if need_even:
                    y += y%2
            else:
                x += 1
                x += x%2

            d = im_count - y*x


        return (x, y)

    def get_plt_im_characteristics(self):

        if self._im_original is not None:
            im_count = 5
        if self._im_debanded is not None:
            im_count += 1
        if self._im_leveled is not None:
            im_count += 1

        plt_grid = self.get_plts_grid(im_count)
        fig = self.get_fig()

        im_cur_count = 1
        if self._im_original is not None:
            ax = fig.add_subplot(plt_grid[0], plt_grid[1] ,im_cur_count, 
                title="Original image {0}".format(self._im_path))
            ax.imshow(self._im_original)
            im_cur_count += 1
        if self._im_debanded is not None:
            ax = fig.add_subplot(plt_grid[0], plt_grid[1] ,im_cur_count, 
                title="Debanded image")
            ax.imshow(self._im_debanded)
            im_cur_count += 1
        if self._im_leveled is not None:
            ax = fig.add_subplot(plt_grid[0], plt_grid[1] ,im_cur_count, 
                title="Leveled image")
            ax.imshow(self._im_leveled)
            im_cur_count += 1

        ax = fig.add_subplot(plt_grid[0], plt_grid[1] ,im_cur_count, 
            title="Gray-scale")
        ax.imshow(self.get_gs_im(), cmap = plt.cm.Greys_r)
        im_cur_count += 1

        cur_im = self.get_cur_im()

        ax = fig.add_subplot(plt_grid[0], plt_grid[1] ,im_cur_count, 
            title="Red channel")
        ax.imshow(cur_im[:,:,0], cmap=plt.cm.Greys_r)
        im_cur_count += 1

        ax = fig.add_subplot(plt_grid[0], plt_grid[1] ,im_cur_count, 
            title="Green channel")
        ax.imshow(cur_im[:,:,1], cmap=plt.cm.Greys_r)
        im_cur_count += 1

        ax = fig.add_subplot(plt_grid[0], plt_grid[1] ,im_cur_count, 
            title="Blue channel")
        ax.imshow(cur_im[:,:,0], cmap=plt.cm.Greys_r)
        im_cur_count += 1

        return fig

    def get_plt_free_detect(self):
        #Plotting blob detection
        fig = self.get_fig()
        ax = fig.add_subplot(2,1,1, title="First blob detection step")
        ax.imshow(self._filter_free_detect.filter_array)
        ax = fig.add_subplot(2,1,2, title="All individual blobs at once")
        ax.imshow(self._filter_array)
        for k, v in self._filter_c_o_m.items():
            ax.text(v[1], v[0], str(k), bbox=dict(facecolor='white', alpha=0.5))

        return fig

    def get_plt_grid(self, grid=None, title="Grid result"):

        fig = self.get_fig()
        if grid is None:
            grid = self._grid

        ax = fig.add_subplot(1,1,1, title=title)
        ax.imshow(self.get_cur_im())

        col_span = (grid[1].min(), grid[1].max())
        row_span = (grid[0].min(), grid[0].max())

        for column in grid[0]:
            line = plt.Line2D((column, column), col_span)
            ax.add_line(line)

        for row in grid[1]:
            line = plt.Line2D(row_span, (row, row))
            ax.add_line(line)

        return fig


    def get_plt_features_detected(self):

        features_with_filter = [f for f in self._features if f.get_has_filter()]
        plt_grid = self.get_plts_grid(len(features_with_filter)*2, need_even=True)

        fig = self.get_fig()
        for i, f in enumerate(features_with_filter):
            ax = fig.add_subplot(plt_grid[0], plt_grid[1], 2*i + 1)
            ax.set_title(f.get_identifier(), fontsize='xx-small')
            ax.imshow(f.get_filter(), cmap=plt.cm.Greys_r)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = fig.add_subplot(plt_grid[0], plt_grid[1], 2*i + 2)
            ax.imshow(self.get_gs_im(rect=f.get_rect()), cmap=plt.cm.Greys_r)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        return fig

    def get_plt_features(self, features=None, f=None, title="", dist=10):
        """
            places text labels in nice positions produced as the results of:

            f   should take the argument as f(features[x][2])
            --  the contents of features[x][2]

            returns a figure
        """
        if title == "":
            title = self._im_path
        if features is None:
            features = [(feat.get_identifier(), feat.get_center(), feat.get_rgb_str()) \
                for feat in self._features]

        fig = self.get_fig()
        ax = fig.add_subplot(1,1,1, title=title)
        ax.imshow(self.get_cur_im())
        for feat in features:
            if f is not None:
                ax.text(feat[1][1]+dist, feat[1][0]-dist, 
                    str(f(feat[2])), 
                    bbox=dict(facecolor='white', alpha=0.5))
            else:
                ax.text(feat[1][1]+dist, feat[1][0]-dist, 
                    str(feat[2]), 
                    bbox=dict(facecolor='white', alpha=0.5))

        return fig

    def get_plt_feature_by_identifier(self, identifier):

        f = self.get_feature_by_identifier(identifier)

        if f:
            plts = 1 + f.get_has_filter()
            fig = self.get_fig()
            ax = fig.add_subplot(plts,1,1, title='Feature {0}'.format(f.get_identifier()))
            ax.imshow(self.get_im_section(rect=f.get_rect()))
            if plts == 2:
                ax = fig.add_subplot(2,1,2, title='Filter {0}'.format(f.get_identifier()))
                ax.imshow(f.get_filter(), cmap=plt.cm.Greys_r)

            return fig

        else:
            return None

    def get_path(self):

        return self._im_path

    def get_rgbs(self, normed=False, pos=False, neg=False, tests=None):

        if normed:
            get_rgb = lambda x: x.get_rgb_unit_vector()
        else:
            get_rgb = lambda x: x.get_rgb_vector()

        return_list = []
        for f in self._features:
            if tests == True:
                if f.get_is_ctrl() == False:
                    return_list.append(get_rgb(f))
            elif tests == False:
                if f.get_is_ctrl() == True:
                    return_list.append(get_rgb(f))
            elif pos:
                if f.get_is_pos_ctrl() == True:
                    return_list.append(get_rgb(f))
            elif neg:
                if f.get_is_neg_ctrl() == True:
                    return_list.append(get_rgb(f))
            else:
                return_list.append(get_rgb(f))

        return return_list

    def set_free_detect(self):

        im_all_blobs = cell.Blob(self._parent, self._im_path,
                self.get_gs_im(), 
                run_detect=False, threshold=None,
                use_fallback_detection=False, image_color_logic = "inv",
                center=None, radius=None)

        im_all_blobs.set_first_step_filtering()
        blobs, qualities, c_o_m, filter_array = im_all_blobs.get_candidate_blob_ranks()

        self._filter_free_detect = im_all_blobs
        self._filter_array = filter_array
        self._filter_blobs = blobs
        self._filter_qualities = qualities
        self._filter_c_o_m = c_o_m

    def set_features_from_rects(self, rects, append=True, settings = {}):

        if not append:
            self.features = []

        for i, rect in enumerate(rects):
            settings['identifier'] = str(self._next_feature_identifier)
            f = Feature(self, rect=rect, settings=settings)
            self._features.append(f)
            self._next_feature_identifier += 1

    def set_grid(self, interactive=True):

        self.set_free_detect()

        g = grid.Grid_Analysis(self._parent)

        im_d0_s, im_d0_f = g.get_spikes(0, self._filter_array>0,
            use_otsu=False, manual_threshold=0.05)
        im_d1_s, im_d1_f = g.get_spikes(1, self._filter_array>0,
            use_otsu=False, manual_threshold=0.05)

        im_f0 = r_signal.get_signal_frequency(im_d0_s)
        im_f1 = r_signal.get_signal_frequency(im_d1_s)

        im_grid = [None, None]

        im_grid[0] = r_signal.get_true_signal(self._filter_array.shape[1], 
            self._pinning_matrix[0] ,
            im_d0_s, frequency =  im_f0, offset_buffer_fraction=0.5)
        im_grid[1] = r_signal.get_true_signal(self._filter_array.shape[0], 
            self._pinning_matrix[1] , 
            im_d1_s, frequency =  im_f1, offset_buffer_fraction=0.5)
        #im3_grid = g.get_analysis(get_gs_im(im3_cropped), pinning_matrix, use_otsu = True,
                #median_coeff=None, verboise=False, visual=False,
                #history=[])
        self._grid = im_grid 
        self._grid_cell_size = (im_f0, im_f1)

        self.set_grid_move(col_move=0.5, row_move=0.5)

        if interactive:
            self.get_plt_grid(grid=self._grid).show()
            row_move = raw_input("In terms of steps, many rows should rows be moved? ")
            col_move = raw_input("And columns? ")
            try:
                row_move = float(row_move)
            except:
                row_move = 0
            try:
                col_move = float(col_move)
            except:
                col_move = 0

            self.set_grid_move(col_move=col_move, row_move=row_move)
            self.get_plt_grid(grid=self._grid, title="Gridding as returned").show()


    def set_grid_move(self, col_move=0, row_move=0):
        self._grid[0] = np.array(r_signal.move_signal([list(self._grid[0])],[col_move], freq_offset=0)[0])
        self._grid[1] = np.array(r_signal.move_signal([list(self._grid[1])],[row_move], freq_offset=0)[0])

    def set_features_from_grid(self, use_cell_fraction=0.7, verbose=False):

        if not self._grid:
            self.set_grid()

        if verbose:
            start_time = time()

        gcs = (self._grid_cell_size[0]*0.7, self._grid_cell_size[1]*0.7)
        
        for r_i, row in enumerate(self._grid[0]):
            for c_i, column in enumerate(self._grid[1]):
                settings = {'center': (column, row),
                    'identifier': "{0}:{1}".format(c_i, r_i),
                    'rect-size': gcs,
                   }
                f = Feature(self, None, settings=settings)
 
                blob = cell.Blob(self._parent, f.get_identifier(), 
                        self.get_gs_im(f.get_rect()),
                        run_detect=True, threshold=None,
                        use_fallback_detection=False, image_color_logic = "inv",
                        center=None, radius=None)
                q_inv = blob.get_candidate_blob_ranks()[1][0]
                if q_inv > 900:
                    self.logger.warning('Colony {0}'.format(settings['identifier']) +\
                        ' had bad blob detection quality, testing inverted logic')
                    blob2 = cell.Blob(self._parent, f.get_identifier(), 
                            self.get_gs_im(f.get_rect()),
                            run_detect=True, threshold=None,
                            use_fallback_detection=False, image_color_logic = None,
                            center=None, radius=None)
                    q_2 = blob2.get_candidate_blob_ranks()[1][0]
                    self.logger.info('Colony {0}'.format(settings['identifier']) +\
                        " got qualities {0} and {1}".format(q_inv, q_2))

                    if q_2 > 1000:
                        self._untrusted.append(settings['identifier'])
                        self.logger.warning('Colony {0}'.format(settings['identifier']) +\
                            " has no good blob. I put it among the untrusted.")
                if q_inv <= 900 or q_inv < q_2:    
                    f.set_filter(blob.filter_array)
                    f.set_quality(q_inv)
                else:
                    f.set_filter(blob2.filter_array)
                    f.set_quality(q_2)

                self._features.append(f)

            if verbose:
                self.logger.info('Got {0:.2f}% of colonies analysed'\
                    .format(100.0*(r_i+1)*(c_i+1)/(len(self._grid[0]) * len(self._grid[1]))))

        if verbose:
            self.logger.info('Complete (run-time {0:.2f} minutes).'\
                .format((time()-start_time)/60.0))

    def set_controls(self, identifier_list, control_types):

        for i, identifier in enumerate(identifier_list):
            f = self.get_feature_by_identifier(identifier)
            if f:
                if control_types[i] is True:
                    f.set_pos_ctrl()
                elif control_types[i] is False:
                    f.set_neg_ctrl()
                elif 'P' in control_types[i].upper():
                    f.set_pos_ctrl()
                else:
                    f.set_neg_ctrl()

    def set_reference(self, identifier):
        f = self.get_feature_by_identifier(identifier)
        if f:
            f.set_ref_ctrl()
            return True
        return False

    def set_references_from_list(self, identifiers):
        no_errors = True
        for identifier in identifiers:
            if not self.set_reference(identifier):
                no_errors = False

        return no_errors


    def set_not_control(self, identifier):

        f = self.get_feature_by_identifier(identifier)
        if f:
            f.set_is_test()
            return True
        
        return False

    def del_feature(self, identifier):

        for i, f in enumerate(self._features):

            if f.get_identifier() == identifier:
                del self._features[i]
                return True

        return False

    def del_untrusted(self):

        for f in self._untrusted:
            self.del_feature(f)
            
    def get_feature_by_identifier(self, identifier):

        try:
            return [f for f in self._features if f.get_identifier() == identifier][0]
        except IndexError:
            return None

    def get_pos_features(self):

        return [f for f in self._features if f.get_is_pos_ctrl()]

    def get_neg_features(self):

        return [f for f in self._features if f.get_is_neg_ctrl()]

    def get_ref_features(self):

        return [f for f in self._features if f.get_is_ref()]

    def get_untrusted_features(self):

        return [f for f in self._features if f.get_identifier() in self._untrusted]

    def get_phenotypes(self, length_normed=True, expected_ref_mean=0.5):

        if length_normed:
            get_vector = lambda x: x.get_rgb_unit_vector()
        else:
            get_vector = lambda x: x.get_rgb_vector()

        pos_vector = np.array([get_vector(f) for f in self.get_pos_features()]).mean(0)
        neg_vector = np.array([get_vector(f) for f in self.get_neg_features()]).mean(0)
        ref_names = [f.get_identifier() for f in self.get_ref_features()]
 
        neg_pos_vector = pos_vector - neg_vector
        pos_distance = np.sqrt((neg_pos_vector**2).sum())

        phenotypes = [] 
        for f in self._features:

            neg_data_vector = get_vector(f) - neg_vector

            phenotypes.append([f.get_identifier(),
                f.get_center(),
                (neg_data_vector * neg_pos_vector).sum() / pos_distance**2,
                f.get_quality()] )

        if len(ref_names) > 0:

            ref_phenotypes = [p[2] for p in phenotypes if p[0] in ref_names]
            ref_mean = np.array(ref_phenotypes).mean()

            ref_coeff = expected_ref_mean / ref_mean

            for i, p in enumerate(phenotypes):

                phenotypes[i][2] = p[2] * ref_coeff

        return phenotypes

class Feature():

    def __init__(self, parent, rect, settings={}):

        self._parent = parent
        self._rgb_vector = None
        self._rgb_unit_vector = None
        self._quality = None

        self._rect = rect
        self._parent.logger.debug("FEATURE: starts with settings {0}".format(settings))

        if 'center' in settings.keys():
            self._center = settings['center']
        else:
            self._center = None

        if 'identifier' in settings.keys():       
            self._identifier = settings['identifier']
        else:
            self._identifier = None

        if 'filter' in settings.keys():
            self._filter_array = settings['filter']
        else:
            self._filter_array = None

        if 'control' in settings.keys() and 'control-type' in settings.keys():
            self._control = True
            if settings['control-type'] == True:
                self._control_type = "Pos"
            elif settings['control-type'] == False:
                self._control_type = "Neg"
            elif 'P' in settings['control-type'].upper():
                self._control_type = "Pos"
            elif 'R' in settings['control-type'].upper():
                self._control_type = "Ref"
            else:
                self._control_type = "Neg"
        else: 
            self._control = False
            self._control_type = None

        if self._rect is None and 'rect-size' in settings.keys():
            gcs = [x/2.0 for x in settings['rect-size']]
            self._rect = [[self._center[0]-gcs[0],self._center[0]+gcs[0]],
                [self._center[1]-gcs[1],self._center[1]+gcs[1]]]

    def get_has_filter(self):
        if self._filter_array is None:
            return False
        else:
            return True

    def get_rect(self):
        return self._rect

    def get_center(self):
        if not self._center:
            self.set_center()

        return self._center

    def get_rgb_vector(self):
        if self._rgb_vector is None:
            self.set_rgb_vector()

        return self._rgb_vector

    def get_rgb_unit_vector(self):
        if self._rgb_unit_vector is None:
            self.set_rgb_unit_vector()

        return self._rgb_unit_vector

    def get_is_pos_ctrl(self):

        return self._control and self._control_type == "Pos"

    def get_is_ref(self):

        return self._control and self._control_type == "Ref"

    def get_is_neg_ctrl(self):
        return self._control and self._control_type == "Neg"

    def get_is_ctrl(self):
        return self._control

    def get_rgb_str(self):

        l = list(self.get_rgb_vector())

        return str(map(round, l))[1:-1].replace(".0","").replace(",","")

    def get_length(self):

        return np.sqrt(self._rgb_vector.dot(self._rgb_vector))

    def get_identifier(self):
        return self._identifier

    def set_center(self):

        self._center = (sum(self._rect[0])/2.0, sum(self._rect[1])/2.0)

    def get_filter(self):

        return self._filter_array

    def get_quality(self):

        return self._quality

    def set_filter(self, filter_array):

        #if self._rect[0] == filter_array.shape[0] and\
        #    self._rect[1] == filter_array.shape[1]:

        self._filter_array = filter_array

    def set_quality(self, quality):

        self._quality = quality

    def _set_is_ctrl(self, ctrl_type="Neg"):
        self._control = True
        self._control_type = ctrl_type

    def set_pos_ctrl(self):
        self._set_is_ctrl(ctrl_type="Pos")

    def set_neg_ctrl(self):
        self._set_is_ctrl(ctrl_type="Neg")

    def set_ref_ctrl(self):
        self._set_is_ctrl(ctrl_type="Ref")

    def set_is_test(self):
        self._control = None
        self._control_type = None

    def set_rgb_vector(self):

        if self._filter_array is not None:
            self.set_rgb_from_filter() 
        else:
            self._rgb_vector = self._parent.get_im_section(rect=self._rect).mean(0).mean(0)

    def set_rgb_unit_vector(self):
        if self._rgb_vector is None:
            self.set_rgb_vector()

        self._rgb_unit_vector = self._rgb_vector / self.get_length()

    def set_rgb_from_filter(self):

        im = self._parent.get_cur_im()
        im_section = im[self._center[0] - self._filter_array.shape[0] /2.0 :\
            self._center[0] + self._filter_array.shape[0] / 2.0,
            self._center[1] - self._filter_array.shape[1] / 2.0 :\
            self._center[1] + self._filter_array.shape[1] / 2.0]

        self._rgb_vector = im_section[np.where(self._filter_array>0)].mean(0)



