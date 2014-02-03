#!/usr/bin/env python
"""The Calibration Controller"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import numpy as np

#
# INTERNAL DEPENDENCIES
#

import model_calibration
import view_calibration

import src.gui.generic.view_generic as view_generic
import src.gui.generic.controller_generic as controller_generic

import src.resource_fixture_image as resource_fixture_image
import src.resource_scanner as resource_scanner
import src.resource_grayscale as resource_grayscale
import src.resource_image as resource_image
import scanomatic.io.logger as logger

#
# EXCEPTIONS
#


class Bad_Stage_Call(Exception):
    pass


class No_View_Loaded(Exception):
    pass


class Not_Yet_Implemented(Exception):
    pass


class Impossible_Fixture(Exception):
    pass


class UnDocumented_Error(Exception):
    pass


class Scan_Failed(Exception):
    pass

#
# CLASSES
#


class Calibration_Controller(controller_generic.Controller):

    def __init__(self, main_controller):

        super(Calibration_Controller, self).__init__(main_controller)

        self._logger = logger.Logger("Calibration Controller")
        self._specific_controller = None

    def _get_default_view(self):

        return view_calibration.Calibration_View(self, self._model)

    def _get_default_model(self):

        tc = self.get_top_controller()
        return model_calibration.get_gui_model(tc.paths)

    def set_mode(self, widget, calibration_mode):

        view = self._view
        model = self._model

        if calibration_mode == 'fixture':

            self._specific_controller = Fixture_Controller(
                self, model=model, view=view)

        elif calibration_mode == "grayscale":

            self._specific_controller = Grayscale_Controller(
                self, model=model, view=view)

        elif calibration_mode == "poly":

            err = Not_Yet_Implemented("Mode 'Cell Count Calibration'")

            raise err
            return

        else:

            raise Bad_Stage_Call(calibration_mode)
            return

        self.add_subcontroller(self._specific_controller)


class Grayscale_Controller(controller_generic.Controller):

    def __init__(self, parent, view=None, model=None,
                 specific_model=None):

        super(Grayscale_Controller, self).__init__(
            parent, view=view, model=model)

        self._logger = logger.Logger("Grayscale Controller")
        tc = self.get_top_controller()
        self._paths = tc.paths

        #MODEL
        if specific_model is not None:
            self._specific_model = specific_model
        else:
            self.build_new_specific_model()

        #VIEW
        view.set_controller(self)
        top = view_calibration.Grayscale_Top(self, model, self._specific_model)
        self._stage = view_calibration.Grayscale_Stage(self, model,
                                                       self._specific_model)
        view.set_top(top)
        view.set_stage(self._stage)

    def build_new_specific_model(self):

        sm = model_calibration.copy_model(
            model_calibration.specific_grayscale_model)
        self._specific_model = sm
        return sm

    def loadImage(self, *args):

        file_path = view_generic.select_file(
            self._model['fixture-calibration-select-im'],
            multiple_files=False, file_filter=
            self._model['fixture-image-file-filter'],
            start_in=self._paths.experiment_root)
        if file_path is not None and len(file_path) > 0:
            sm = self._specific_model
            sm['im-path'] = file_path[0]
            sm['source-sourceValues'] = None
            sm['source-targetValues'] = None
            sm['target-sourceValues'] = None
            sm['target-targetValues'] = None
            self.setActiveGrayscale(isSource=sm['active-type'] == 'source')
            self._stage.updateImage()

    def setSourceGrayscale(self, widget):

        sm = self._specific_model
        sm['source-name'] = widget.get_text()
        sm['source-targetValues'] = None
        sm['source-sourceValues'] = None
        self._stage.clearOverlay('source', doRedraw=True)

    def setTargetGrayscale(self, widget):

        sm = self._specific_model
        sm['target-name'] = widget.get_text()
        sm['target-targetValues'] = None
        sm['target-sourceValues'] = None
        self._stage.clearOverlay('target', doRedraw=True)

    def setActiveGrayscale(self, isSource):

        sm = self._specific_model
        m = self._model
        sm['active-type'] = m['grayscale-types'][isSource]
        sm['active-color'] = m['grayscale-colors'][isSource]
        sm['active-target'] = None
        sm['active-source'] = None
        self._stage.updateActiveSeletion()

    def setGrayScaleAreaSource(self, x, y):

        sm = self._specific_model

        if sm['active-type'] is not None:
            if x is not None and y is not None:
                sm['active-source'] = (x, y)
                sm['active-changing'] = True
            else:
                sm['active-source'] = None
                sm['active-changing'] = False

        self._stage.updateActiveSeletion()

    def getSlicer(self, pos1, pos2):

        return [slice(*dimension) for dimension in
                map(sorted, zip(pos1, pos2))][::-1]

    def setGrayScaleAreaTarget(self, x, y):

        sm = self._specific_model

        if sm['active-type'] is not None:
            if x is not None and y is not None:

                sm['active-target'] = (x, y)
                sm['active-changing'] = False
                if sm['active-type'] == 'source':
                    gsType = sm['source-name']
                else:
                    gsType = sm['target-name']

                gsAnalysis = resource_image.Analyse_Grayscale(
                    gsType, sm['im'][self.getSlicer(sm['active-target'],
                                                    sm['active-source'])])

                if sm['active-type'] == 'source':

                    sm['source-sourceValues'] = gsAnalysis.get_source_values()
                    sm['source-targetValues'] = gsAnalysis.get_target_values()
                    if sm['source-sourceValues'] is not None:
                        sm['source-polynomial'] = np.poly1d(np.polyfit(
                            sm['source-sourceValues'],
                            sm['source-targetValues'], 3))
                    else:
                        sm['source-polynomial'] = None

                    self._stage.updateActiveSeletion(gsAnalysis=gsAnalysis)

                else:

                    if (sm['source-polynomial'] is not None and
                            sm['target-sourceValues'] is not None):
                        sm['target-sourceValues'] = gsAnalysis.get_source_values()

                        sm['target-targetValues'] = sm['source-polynomial'](
                            [np.mean(s) for s in gsAnalysis.slices])

                        self._stage.setAllowSave(
                            True, message=self._model['grayscale-info-done'])

                    else:

                        sm['target-sourceValues'] = None
                        sm['target-targetValues'] = None
                        self._stage.setAllowSave(False)

                    """This is probably wrong
                    if (sm['source-polynomial'] is not None and
                            sm['target-sourceValues'] is not None):

                        sm['target-targetValues'] = sm['source-polynomial'](
                            sm['target-sourceValues'])

                        self._stage.setAllowSave(
                            True, message=self._model['grayscale-info-done'])

                    else:

                        sm['target-sourceValues'] = None
                        sm['target-targetValues'] = None
                        self._stage.setAllowSave(False)
                    """
                    self._stage.updateActiveSeletion(
                        gsAnalysis=gsAnalysis,
                        targetValues=sm['target-targetValues'])

    def saveNewGrayscale(self, *args):

        resource_grayscale.updateGrayscaleValues(
            self._specific_model['target-name'],
            targets=list(self._specific_model['target-targetValues']))

        self._stage.setAllowSave(
            False, message=self._model['grayscale-info-saved'])


class Fixture_Controller(controller_generic.Controller):

    def __init__(self, parent, view=None, model=None, specific_model=None):

        super(Fixture_Controller, self).__init__(
            parent, view=view, model=model)

        self._logger = logger.Logger("Fixture Controller")

        tc = self.get_top_controller()
        self._paths = tc.paths
        self._config = tc.config
        self._scanners = resource_scanner.Scanners(self._paths, self._config)

        self._window = self.get_window()

        #MODEL
        if specific_model is not None:
            self._specific_model = specific_model
        else:
            self.build_new_specific_model()

        #VIEW
        view.set_controller(self)
        self.set_view_stage(None, 'fixture-select')

    def set_view_stage(self, widget, stage_call, *args, **kwargs):

        if len(args) > 0:
            sm = args[0]
        else:
            sm = self._specific_model

        m = self._model
        view = self._view

        if stage_call == 'fixture-select':

            top = view_calibration.Fixture_Select_Top(self, m, sm)
            stage = view_calibration.Fixture_Select_Stage(self, m, sm)

        elif stage_call == 'marker-calibration':

            self.set_unsaved()
            self.get_top_controller().fixtures.fill_model(sm)

            top = view_calibration.Fixture_Marker_Calibration_Top(self, m, sm)

            if len(sm['marker-positions']) >= 3:
                top.set_allow_next(True)

            stage = view_calibration.Fixture_Marker_Calibration_Stage(
                self, m, sm)

        elif stage_call == 'segmentation':

            top = view_calibration.Fixture_Segmentation_Top(self, m, sm)

            stage = view_calibration.Fixture_Segmentation_Stage(self, m, sm)

        elif stage_call == 'save':

            top = view_calibration.Fixture_Save_Top(self, m, sm)

            stage = view_calibration.Fixture_Save_Stage(self, m, sm)

            self.save_fixture()

        else:

            err = Bad_Stage_Call("{0} recieved call '{1}' from {2}".format(
                self, stage_call, widget))

            raise err

        view.set_top(top)
        view.set_stage(stage)

    def build_new_specific_model(self):

        sm = model_calibration.copy_model(
            model_calibration.specific_fixture_model)
        self._specific_model = sm
        return sm

    def check_fixture_select(self, widget, is_new):

        stage = self._view.get_stage()
        top = self._view.get_top()
        sm = self._specific_model

        if is_new:

            new_name = stage.new_name.get_text()
            if new_name == "" or new_name in \
                    self.get_top_controller().fixtures.get_names():

                warn = True
                allow_next = False

            else:

                warn = False
                allow_next = True

            sm['fixture'] = new_name
            sm['new_fixture'] = True

        else:

            treemodel, rows = stage.selection.get_selected_rows()
            allow_next = len(rows) > 0
            warn = None
            if allow_next:
                sm['fixture'] = treemodel[rows[0]][0]
            sm['new_fixture'] = False

        stage.set_bad_name_warning(warn)
        top.set_allow_next(allow_next)

    def set_image_path(self, widget):

        image_list = view_generic.select_file(
            self._model['fixture-image-dialog'],
            multiple_files=False,
            file_filter=self._model['fixture-image-file-filter'],
            start_in=self.get_top_controller().paths.experiment_root)

        if len(image_list) > 0:

            self._set_new_image(image_list[0])

    def _set_new_image(self, im_path):

        sm = self._specific_model
        sm['im-path'] = im_path
        sm['marker-positions'] = list()
        sm['im-scale'] = 1.0
        sm['im-original-scale'] = 1.0

        stage = self.get_view().get_stage()
        stage.set_new_image()
        stage.check_allow_marker_detection()

    def set_image_scan(self, widget):

        m = self._model

        scanner_name = view_generic.claim_a_scanner_dialog(
            self._window, m['scan-fixture-text'], self._paths.martin,
            self._scanners)

        if scanner_name is not None:
            scanner = self._scanners[scanner_name]
            scanner.claim()
            if scanner.scan("TPU", self._paths.fixture_tmp_scan_image):
                scanner.free()
                self._set_new_image(self._paths.fixture_tmp_scan_image)

            else:
                scanner.free()
                raise Scan_Failed()

    def handle_keypress(self, widget, event):

        sm = self._specific_model

        if sm['active-segment'] is not None:

            if view_calibration.gtk.gdk.keyval_name(event.keyval) == "Delete":

                if sm['active-segment'] == 'G':

                    sm['grayscale-coords'] = list()
                    sm['grayscale-sources'] = None

                else:

                    try:
                        plate = int(sm['active-segment'][-1]) - 1
                    except:
                        plate = None

                    if plate is not None:
                        sm['plate-coords'][plate] = None

            self._view.get_stage().update_segment(
                sm['active-segment'],
                scale=sm['im-original-scale'])

    def save_fixture(self):

        sm = self._specific_model
        self.f_settings['history'].reset_all_gridding_histories()

        for plate in enumerate(sm['plate-coords']):
            self.f_settings['plate-coords'] = plate

        self._logger.info(
            "The fixture has {0} grayscale with source values {1}".format(
                sm['grayscale-type'], sm['grayscale-sources']))

        #self.f_settings['grayscale'] = sm['grayscale-sources']
        self.f_settings['current'].save()
        self.set_saved()
        self.get_top_controller().fixtures.update()

    def set_number_of_markers(self, widget):

        s = widget.get_text()

        if s != "":

            try:

                n = int(widget.get_text())

            except:

                n = 0
                widget.set_text(str(n))

        else:

            n = 0

        self._specific_model['markers'] = n
        self._view.get_stage().check_allow_marker_detection()

    def run_marker_detect(self, widget):

        sm = self._specific_model

        if sm['markers'] >= 3 and sm['im'] is not None:

            if sm['marker-path'] is None:
                sm['marker-path'] = self.get_top_controller().paths.marker

            self.f_settings = resource_fixture_image.Fixture_Image(
                sm['fixture-file'],
                fixture_directory=self.get_top_controller().paths.fixtures,
                image_path=sm['im-path'],
                im_scale=(sm['im-scale'] < 1 and sm['im-scale'] or None),
                define_reference=True, markings_path=sm['marker-path'],
                markings=sm['markers'])

            self.f_settings.run_marker_analysis()
            X, Y = self.f_settings['markers']

            if X is not None and Y is not None:

                sm['marker-positions'] = zip(X, Y)

            else:

                sm['marker-positions'] = None

            self._view.get_stage().set_markers()

            if len(sm['marker-positions']) >= 3:
                self._view.get_top().set_allow_next(True)

        else:

            err = Impossible_Fixture(
                "Markers must be at least 3 (you have '{0}')".format(
                sm['markers']) + " and you need an image too")

            raise err

        if len(sm['marker-positions']) > 3:
            self._view.get_top().set_allow_next(True)

    def toggle_grayscale(self, widget):

        has_gs = widget.get_active()
        stage = self._view.get_stage()
        sm = self._specific_model
        sm['grayscale-exists'] = has_gs
        if has_gs is False:
            sm['grayscale-coords'] = list()
            sm['grayscale-sources'] = None
            stage.update_segment(
                'G', scale=sm['im-original-scale'])

        stage.set_segments_in_list()

    def setGrayscaleType(self, widget, data=None):

        gsName = widget.get_text()
        sm = self._specific_model
        sm['grayscale-type'] = gsName
        self.f_settings['grayscale_type'] = gsName
        self.get_grayscale()
        self._view.get_stage().set_segments_in_list()

    def set_number_of_plates(self, widget):

        try:
            plates = int(widget.get_text())
        except:
            plates = 0
            if widget.get_text() != "":
                widget.set_text("{0}".format(plates))

        sm = self._specific_model

        stage = self._view.get_stage()
        old_number_of_plates = len(sm['plate-coords'])
        for p in xrange(plates, old_number_of_plates):
            sm['plate-coords'][p] = None
            stage.update_segment("P{0}".format(p + 1))
        sm['plate-coords'] += [None] * (plates - len(sm['plate-coords']))
        sm['plate-coords'] = sm['plate-coords'][:plates]

        self._view.get_stage().set_segments_in_list()

    def set_active_segment(self, selection):

        store, rows = selection.get_selected_rows()
        sm = self._specific_model

        if len(rows) == 0:

            sm['active-segment'] = None

        else:

            row = rows[0]
            sm['active-segment'] = store[row][2]

    def set_allow_save(self, val):

        if val and len(self._specific_model['plate-coords']) > 0:
            self._view.get_top().set_allow_next(True)
        else:
            self._view.get_top().set_allow_next(False)

    def get_segment_ok(self, pos1, pos2):

        if pos1 is None or None in pos1 or pos2 is None or None in pos2:

            return None

        return zip(*map(sorted, zip(*(pos1, pos2))))

    def get_grayscale(self):

        sm = self._specific_model

        if sm['grayscale-coords'] is None or len(sm['grayscale-coords']) != 2:

            gs_ok = False

        else:

            self.f_settings['grayscale-coords'] = sm['grayscale-coords']

            self.f_settings.analyse_grayscale()
            gs_target = self.f_settings['grayscaleTarget']
            gs_source = self.f_settings['grayscaleSource']
            gs_source = np.array(gs_source)
            gs_target = np.array(gs_target)

            np.save("tmp_gs_{0}_source.npy".format(self.f_settings['name']),
                    gs_source)
            np.save("tmp_gs_{0}_target.npy".format(self.f_settings['name']),
                    gs_target)
            if resource_grayscale.validate(self.f_settings):

                sm['grayscale-sources'] = gs_source
                gs_ok = True

            else:

                gs_ok = False

        if gs_ok is False:

            sm['grayscale-sources'] = None

        return gs_ok

    def mouse_press(self, event, *args, **kwargs):

        sm = self._specific_model
        scale = sm['im-original-scale']

        if sm['active-segment'] is not None:
            if event.xdata is not None and event.ydata is not None:
                sm['active-source'] = (event.xdata / scale,
                                       event.ydata / scale)
            else:
                sm['active-source'] = None

    def mouse_release(self, event, *args, **kwargs):

        sm = self._specific_model
        scale = sm['im-original-scale']

        if sm['active-segment'] is not None:
            if event.xdata is not None and event.ydata is not None:

                sm['active-target'] = (event.xdata / scale,
                                       event.ydata / scale)

            coords = self.get_segment_ok(
                sm['active-source'],
                sm['active-target'])

            if coords is not None:

                if sm['active-segment'] == "G":

                    sm['grayscale-coords'] = coords
                    self.get_grayscale()
                    self._view.get_stage().draw_active_segment(
                        scale=scale)

                else:
                    try:

                        plate = int(sm['active-segment'][-1]) - 1
                    except:
                        plate = None

                    if plate is not None:
                        self.set_plates(plate, coords)
                        #sm['plate-coords'][plate] = coords

                    self._view.get_stage().draw_all_plates(scale=scale)

        sm['active-source'] = None
        sm['active-target'] = None

    def mouse_move(self, event, *args, **kwargs):

        sm = self._specific_model
        scale = sm['im-original-scale']

        if sm['active-segment'] is not None and \
                sm['active-source'] is not None:

            if event.xdata is not None and event.ydata is not None:

                sm['active-target'] = (event.xdata / scale,
                                       event.ydata / scale)

            self._view.get_stage().draw_active_segment(scale=scale)

    def set_plates(self, replace_index, new_coords):

        sm = self._specific_model
        sm['plate-coords'][replace_index] = new_coords

        self._sort_plates()

    def _sort_plates(self):
        """Sorts plate names according to UL -> UR -> LL -> LR scheme"""

        sm = self._specific_model
        plates = sm['plate-coords']

        p_indices = [i for i, p in enumerate(plates) if p is not None]
        p_names = [pn + 1 for pn in range(len(p_indices))]
        p_new_indices = list()

        for pn in p_names:

            if pn == 1:

                tmp_eval = np.array(
                    [p[0][0] + p[0][1] for i, p in enumerate(plates)
                     if i in p_indices])

            elif pn == 2:

                tmp_eval = np.array([p[0][1] for i, p in enumerate(plates) if i in p_indices])

            elif pn == 3:

                tmp_eval = np.array([p[0][0] for i, p in enumerate(plates) if i in p_indices])

            else:

                tmp_eval = np.array([p[0][1] for i, p in enumerate(plates) if i in p_indices])

            p_new_indices.append(p_indices[tmp_eval.argmax()])
            p_indices.remove(p_new_indices[-1])

        new_plates = [None] * 4
        for new_index, old_index in enumerate(p_new_indices):
            new_plates[new_index] = plates[old_index]

        sm['plate-coords'] = new_plates
