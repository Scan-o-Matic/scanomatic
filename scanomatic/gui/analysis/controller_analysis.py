"""The Analysis Controller"""
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

import os
import glob
import re
import gobject
import threading
import numpy as np
import copy
from subprocess import Popen, PIPE

#
# INTERNAL DEPENDENCIES
#

import model_analysis
import view_analysis

#import scanomatic.gui.subprocs.communications.gui_communicator as gui_communicator
import scanomatic.gui.generic.view_generic as view_generic
import scanomatic.gui.generic.controller_generic as controller_generic

import scanomatic.imageAnalysis.wrappers as a_wrapper
import scanomatic.imageAnalysis.imageFixture as imageFixture
import scanomatic.imageAnalysis.imageBasics as imageBasics
import scanomatic.imageAnalysis.imageGrayscale as imageGrayscale
import scanomatic.imageAnalysis.grayscale as grayscale

import scanomatic.io.config_file as config_file
import scanomatic.io.verificationTags as verificationTags
import scanomatic.io.logger as logger
import scanomatic.io.project_log as project_log
import scanomatic.io.paths as paths
#from run_make_project import Make_Project

#
# EXCEPTIONS
#


class Bad_Stage_Call(Exception):
    pass


class No_View_Loaded(Exception):
    pass


class Not_Yet_Implemented(Exception):
    pass


class Unknown_Log_Request(Exception):
    pass


class UnDocumented_Error(Exception):
    pass

#
# CLASSES
#


class Analysis_Controller(controller_generic.Controller):

    def __init__(self, main_controller, **kwargs):

        super(Analysis_Controller, self).__init__(
            main_controller, controller_name='A')

        self._logger = logger.Logger("Analysis Controller")
        self.transparency = Analysis_Transparency_Controller(
            self, view=self._view, model=self._model)

        self._specific_controller = None
        self.fixture = None

        if 'stage' in kwargs:
            self.set_analysis_stage(None, kwargs['stage'], **kwargs)

    def ask_destroy(self, *args, **kwargs):

        if self._specific_controller is not None:
            val = self._specific_controller.ask_destroy(*args, **kwargs)
            if val:
                self.destroy()
            return val
        else:
            return True

    def destroy(self):

        if self._specific_controller is not None:
            self._specific_controller.destroy()

    def _get_default_view(self):

        return view_analysis.Analysis(self, self._model)

    def _get_default_model(self):

        return model_analysis.get_gui_model()

    def _callback(self, user_data):

        if user_data is not None:

            if user_data['view-data'] is None and 'cb' not in user_data:

                user_data['cb'] = 9

            if user_data['thread'].is_alive() is False:

                user_data['view-function'](user_data['view-complete'])
                user_data['view'].run_release()

                if 'complete-function' in user_data:

                    user_data['complete-function']()

                return False

            else:

                if user_data['view-data'] is None:

                    user_data['cb'] += 1
                    user_data['view-function'](1 - np.exp(-0.01 * user_data['cb']))

                else:

                    user_data['view-function'](user_data['view-data'])

        if user_data is None:

            print "LOST!"
            return False

        gobject.timeout_add(587, self._callback, user_data)
        return None

    def _get_safe_slice(self, coords, im_shape):

        coords = list(map(list, coords))

        self._logger.info(
            "Slice coords before boundry check {0}".format(coords))
        if coords[0][0] < 0:
            coords[0][0] = 0
        if coords[0][1] < 0:
            coords[0][1] = 0

        if coords[1][0] >= im_shape[1]:
            coords[1][0] = im_shape[1] - 1
        if coords[1][1] >= im_shape[0]:
            coords[1][1] = im_shape[0] - 1

        self._logger.info(
            "Slice coords after boundry check {0}".format(coords))

        return coords

    def get_available_fixtures(self):

        return self.get_top_controller().fixtures.get_names()

    def set_analysis_stage(self, widget, *args, **kwargs):

        if len(args) < 1:

            raise Bad_Stage_Call()

        else:

            stage_call = args[0]

            view = self.get_view()
            model = self.get_model()

            if view is None:

                raise No_View_Loaded()

            if stage_call == "about":

                view.set_top()
                view.set_stage()

            elif stage_call == "project":

                self._specific_controller = \
                    Analysis_Project_Controller(
                        self,
                        view=self._view, model=self._model,
                        **kwargs)

                self.add_subcontroller(self._specific_controller)

                view.set_top(view_analysis.Analysis_Top_Project(
                    self._specific_controller, model))

                view.set_stage(view_analysis.Analysis_Stage_Project(
                    self._specific_controller, model))

            elif stage_call == "1st_pass":

                self._specific_controller = Analysis_First_Pass(
                    self, view=view, model=self._model)

                self.add_subcontroller(self._specific_controller)

                view.set_top(
                    view_analysis.Analysis_First_Pass_Top(
                        self._specific_controller, model))

                view.set_stage(
                    view_analysis.Analysis_Stage_First_Pass(
                        self._specific_controller, model))

            elif stage_call == "inspect":

                self._specific_controller = Analysis_Inspect(
                    self, view=view, model=self._model,
                    **kwargs)

                self.add_subcontroller(self._specific_controller)

                view.set_top(
                    view_analysis.Analysis_Inspect_Top(
                        self._specific_controller, model))

                view.set_stage(
                    view_analysis.Analysis_Inspect_Stage(
                        self._specific_controller, model))

            elif stage_call == 'convert':

                self._specific_controller = Analysis_Convert(
                    self, view=view, model=self._model, **kwargs)

                self.add_subcontroller(self._specific_controller)
                view.set_top(
                    view_analysis.Analysis_Convert_Top(
                        self._specific_controller, model))
                view.set_stage(
                    view_analysis.Analysis_Convert_Stage(
                        self._specific_controller, model))

            elif stage_call == "extract":

                self._specific_controller = Analysis_Extract(
                    self, view=view, model=self._model, **kwargs)

                self.add_subcontroller(self._specific_controller)
                view.set_top(
                    view_analysis.Analysis_Extract_Top(
                        self._specific_controller, model))
                view.set_stage(
                    view_analysis.Analysis_Extract_Stage(
                        self._specific_controller, model))

            elif stage_call == "transparency":

                #IF CALLED WITHOUT MODEL CREATE ONE
                if len(args) < 2:
                    self.transparency.build_blank_specific_model()

                self.transparency._specific_model['stage'] = 'image-selection'

                view.set_top(view_analysis.Analysis_Top_Image_Selection(
                    self, model,
                    self.transparency.get_specific_model(),
                    self.transparency))

                view.set_stage(view_analysis.Analysis_Stage_Image_Selection(
                    self, model,
                    self.transparency.get_specific_model(),
                    self.transparency))

            elif stage_call == "colour":

                raise Not_Yet_Implemented()

            elif stage_call == "normalisation":

                specific_model = args[1]
                self.set_unsaved()

                if specific_model['mode'] == 'transparency':

                    specific_model['image'] += 1

                    if specific_model['image'] >= len(
                            specific_model['images-list-model']):

                        raise Bad_Stage_Call("Image position overflow")

                    if specific_model['fixture']:

                        specific_model['stage'] = 'auto-calibration'
                        specific_model['plate'] = -1

                        model['fixtures'] = self.get_available_fixtures()

                        top = view_analysis.Analysis_Top_Auto_Norm_and_Section(
                            self, model,
                            specific_model,
                            self.transparency)

                        if specific_model['image'] > 0:
                            message = 'previous-image'
                            label = model[
                                'analysis-top-auto-norm-and-section-prev-image']
                        else:
                            message = 'select-images'
                            label = model[
                                'analysis-top-auto-norm-and-section-sel-images']

                        top.pack_back_button(
                            label,
                            self.transparency.step_back, message)

                        view.set_top(top)

                        view.set_stage(
                            view_analysis.Analysis_Stage_Auto_Norm_and_Section(
                                self, model,
                                specific_model,
                                self.transparency))

                    else:

                        specific_model['stage'] = 'manual-calibration'

                        view.set_top(
                            view_analysis.Analysis_Top_Image_Normalisation(
                                self, model,
                                specific_model,
                                self.transparency))

                        view.set_stage(
                            view_analysis.Analysis_Stage_Image_Norm_Manual(
                                self, model,
                                specific_model,
                                self.transparency))

                elif specific_model['mode'] == 'colour':

                    raise Not_Yet_Implemented((stage_call, specific_model['mode']))

                else:

                    raise Bad_Stage_Call(stage_call)

            elif stage_call == "sectioning":

                self.set_unsaved()
                specific_model = args[1]
                specific_model['stage'] = 'sectioning'
                specific_model['plate-coords'] = list()

                view.set_top(
                    view_analysis.Analysis_Top_Image_Sectioning(
                        self, model,
                        specific_model,
                        self.transparency))

                view.set_stage(
                    view_analysis.Analysis_Stage_Image_Sectioning(
                        self, model,
                        specific_model,
                        self.transparency,
                        self.get_window()))

                specific_model['plate'] = -1

            elif stage_call == "plate":

                self.set_unsaved()

                specific_model = args[1]
                specific_model['plate'] += 1
                specific_model['stage'] = 'plate'

                if self.transparency._log is None:
                    self.transparency._log = Analysis_Log_Controller(
                        self, self._model, specific_model)

                else:

                    self.transparency._log.set_view()
                    self.transparency._log._model['current-strain'] = None

                if specific_model['plate'] < len(specific_model['plate-coords']):

                    coords = self._get_safe_slice(
                        specific_model['plate-coords'][
                            specific_model['plate']],
                        specific_model['image-array'].shape)

                    image = specific_model['image']

                    if image in specific_model['auto-transpose']:

                        specific_model['plate-im-array'] = \
                            specific_model['auto-transpose'][image](
                                specific_model['image-array'][
                                    coords[0][1]: coords[1][1],
                                    coords[0][0]: coords[1][0]])

                        specific_model['plate-is-normed'] = True

                    else:

                        specific_model['plate-im-array'] = \
                            specific_model['image-array'][
                                coords[0][1]: coords[1][1],
                                coords[0][0]: coords[1][0]]

                        specific_model['plate-is-normed'] = False

                    top = view_analysis.Analysis_Top_Image_Plate(
                        self, model,
                        specific_model,
                        self.transparency)

                    if specific_model['plate'] > 0:
                        message = 'previous-plate'
                        label = model[
                            'analysis-top-image-plate-prev_plate']
                    else:
                        message = 'normalisation'
                        label = model[
                            'analysis-top-image-plate-prev_norm']

                    top.pack_back_button(
                        label,
                        self.transparency.step_back, message)

                    view.set_top(top)
                    top.set_allow_next(True)

                    view.set_stage(
                        view_analysis.Analysis_Stage_Image_Plate(
                            self, model,
                            specific_model,
                            self.transparency))

                    view.get_stage().run_lock_select_check()
                    view.get_stage().unset_warning()

            elif stage_call == "log_book":

                specific_model = args[1]
                specific_model['stage'] = 'done'

                view.set_top(
                    view_analysis.Analysis_Top_Done(
                        self, model))

                self.transparency._log.set_view()
                view.set_stage(
                    self.transparency._log.get_view())

            else:

                raise Bad_Stage_Call(stage_call)


class Analysis_Extract(controller_generic.Controller):

    def __init__(self, parent, view=None, model=None, **kwargs):

        super(Analysis_Extract, self).__init__(
            parent, view=view, model=model)

        self._paths = paths.Paths()

        self.set_specific_model(model_analysis.copy_model(
            model_analysis.specific_extract))

    def set_abort(self, *args):

        self._parent().set_analysis_stage(None, "about")

    def test_allow_start(self):

        sm = self._specific_model
        self.get_stage().get_top().set_allow_next(
            os.path.isdir(sm['path']) and sm['tag'] != "" and
            self.get_top_controller().server.connected())

    def check_path(self, path):

        p = self._paths
        return (len(glob.glob(os.path.join(
            path, p.image_analysis_img_data.format("*")))) > 2 and
            os.path.isfile(os.path.join(path, p.image_analysis_time_series)))

    def start(self, *args):

        sm = self._specific_model
        if not self.get_top_controller().server.addExtractionJob(
                sm['path'], sm['tag']):

            self.get_view().get_stage().error(
                self._model['extract-launch-error'])

        else:

            self.destroy()


class Analysis_Convert(controller_generic.Controller):

    def __init__(self, parent, view=None, model=None, **kwargs):

        super(Analysis_Convert, self).__init__(
            parent, view=view, model=model)

    def set_abort(self, *args):

        self._parent().set_analysis_stage(None, "about")

    def start(self, path):

        p = Popen(['scan-o-matic_xml2image_data', '-i', path])
        self.get_view().get_stage().addWorker(p, path)


class Analysis_Inspect(controller_generic.Controller):

    def __init__(self, parent, view=None, model=None, **kwargs):

        super(Analysis_Inspect, self).__init__(
            parent, view=view, model=model)

        self._logger = logger.Logger("Analysis Inspect Controller")
        tc = self.get_top_controller()
        self._paths = tc.paths
        self._app_config = tc.config

        print kwargs
        if 'analysis-run-file' in kwargs:
            gobject.timeout_add(223, self.set_analysis,
                                kwargs['analysis-run-file'],
                                kwargs['project-name'])

        if 'launch-filezilla' in kwargs and kwargs['launch-filezilla']:
            gobject.timeout_add(331, self.launch_filezilla, None)

    def destroy(self):

        sm = self._specific_model
        if sm is not None and 'filezilla' in sm and sm['filezilla']:
            subprocs = self.get_top_controller().subprocs
            subprocs.set_project_progress(
                sm['prefix'], 'UPLOAD', 'COMPLETED')

    def set_analysis(self, run_file, project_name=None):

        self.set_specific_model(model_analysis.copy_model(
            model_analysis.specific_inspect))

        sm = self._specific_model
        sm['run-file'] = run_file
        sm['analysis-dir'] = os.path.dirname(run_file)
        sm['experiment-dir'] = os.path.abspath(os.path.join(
            sm['analysis-dir'], os.path.pardir))

        self._parse_run_file()
        if sm['prefix'] is not None:
            self.get_view().get_stage().set_project_name(sm['prefix'])
        self._look_for_grid_images()
        if (sm['pinnings'] is None or
                sum(sm['pinnings']) !=
                sum([gi is not None for gi in sm['grid-images']])):

            self.get_view().get_stage().set_inconsistency_warning()

        stage = self.get_view().get_stage()
        if project_name:
            stage.set_project_name(project_name)
        stage.set_display(sm)

        subprocs = self.get_top_controller().subprocs
        subprocs.set_project_progress(
            sm['prefix'], 'INSPECT', 'RUNNING')
        subprocs.set_project_progress(
            sm['prefix'], 'UPLOAD', 'LAUNCH')

    def launch_filezilla(self, widget):

        paths = self.get_top_controller().paths
        m = self._model
        sm = self._specific_model
        subprocs = self.get_top_controller().subprocs

        if (Popen('which filezilla', stdout=PIPE,
                  shell=True).communicate()[0] != ""):

            subprocs.set_project_progress(
                sm['prefix'], 'INSPECT', 'COMPLETED')
            subprocs.set_project_progress(
                sm['prefix'], 'UPLOAD', 'RUNNING')
            os.system("filezilla &")
            sm['filezilla'] = True
        else:

            if view_generic.dialog(
                    self.get_window(),
                    self._model['analysis-stage-inspect-upload-install'],
                    'info',
                    yn_buttons=True):

                if os.system('gksu {0} --message="{1}"'.format(
                        paths.install_filezilla,
                        m['analysis-stage-inspect-upload-gksu'])) == 0:

                    subprocs.set_project_progress(
                        sm['prefix'], 'INSPECT', 'COMPLETED')
                    subprocs.set_project_progress(
                        sm['prefix'], 'UPLOAD', 'RUNNING')
                    os.system("filezilla &")
                    sm['filezilla'] = True

                else:

                    view_generic.dialog(
                        self.get_window(),
                        self._model['analysis-stage-inspect-upload-error'],
                        'error',
                        yn_buttons=False)

    def remove_grid(self, plate):

        sm = self._specific_model
        gh = sm['gridding-history']
        failed_remove = False

        if (sm['uuid'] is not None and
                gh is not None and
                sm['pinning-formats'][plate] is not None and
                sm['fixture'] is not None):

            if gh.unset_gridding_parameters(
                    sm['uuid'],
                    sm['pinning-formats'][plate],
                    plate) is not True:

                failed_remove = True

        else:

            failed_remove = True

        if failed_remove:
            self.get_view().get_stage().warn_remove_failed()

        return failed_remove is False

    def _look_for_grid_images(self):

        sm = self._specific_model
        if os.path.isdir(sm['analysis-dir']):
            analysisBase = sm['analysis-dir']
        elif os.path.isdir(os.path.dirname(sm['analysis-run-file'])):
            analysisBase = os.path.dirname(sm['analysis-run-file'])
        else:
            #Are there other ways of finding the stuff?
            analysisBase = sm['analysis-dir']

        im_pattern = os.path.join(
            analysisBase,
            self._paths.experiment_grid_image_pattern)

        sm['grid-images'] = []
        if sm['pinnings'] is not None:

            for i, p in enumerate(sm['pinnings']):

                if p:

                    if os.path.isfile(im_pattern.format(i + 1)):

                        sm['grid-images'].append(im_pattern.format(i + 1))
                    else:
                        sm['grid-images'].append(None)

                else:
                    sm['grid-images'].append(None)

    def _parse_run_file(self):

        sm = self._specific_model
        if sm['run-file'] is not None:
            print "Will look into file", sm['run-file']
            try:
                fh = open(sm['run-file'], 'r')
            except:
                return False

            fh_data = fh.read()
            fh.close()

            experiment_dir = os.path.abspath(os.path.join(
                sm['analysis-dir'], os.pardir))

            #UUID
            p_uuid = re.findall(r"\'UUID\': \'([a-f\d-]*)\'", fh_data)
            if len(p_uuid) > 0:
                sm['uuid'] = p_uuid[0]

            #FIXTURE
            fixture = re.findall(r"\'Fixture\': ([^,]*)", fh_data)
            if len(fixture) > 0:
                fixture = fixture[0]
                if (fixture != 'None' and len(fixture) > 2 and
                        fixture[0] == "'" and fixture[-1] == "'"):

                    sm['fixture'] = fixture[1:-1]  # Trim the single quoutes

            #CHECK FOR FIXTURE NAME IN LOCAL FIXTURE COPY
            if sm['fixture'] is None:
                fixture = imageFixture.Fixture_Image(
                    self._paths.experiment_local_fixturename,
                    fixture_directory=experiment_dir)

                sm['fixture'] = fixture['fixture']['name']

            #LOAD GRIDDING HISTORY
            if sm['fixture'] is not None:

                sm['gridding-history'] = imageFixture.Gridding_History(
                    self,
                    sm['fixture'], self._paths,
                    app_config=self._app_config)

            #PREFIX
            prefix = re.findall(r"\'Prefix\': ([^,]*)", fh_data)
            if len(prefix) > 0:
                prefix = prefix[0]
                if (prefix != 'None' and len(prefix) > 2 and
                        prefix[0] == "'" and prefix[-1] == "'"):

                    sm['prefix'] = prefix[1:-1]

            #PINNING
            pinnings = re.findall(r"\'Pinning Matrices\': ([^']*)", fh_data)
            if len(pinnings) > 0:
                pinnings = pinnings[0]
                try:
                    pinnings = eval(pinnings[:-2])
                    sm['pinnings'] = [p is not None for p in pinnings]
                    sm['pinning-formats'] = pinnings
                except:
                    pass

            #DESCRIPTION
            description = re.findall(r"\'Description\': \'([^']*)\',", fh_data)
            sm['plate-names'] = [''] * len(sm['pinnings'])
            if len(description) > 0:
                plates = re.findall("Plate +(\d) +\"([^\"]*)", description[0])
                for i, name in plates:
                    try:
                        i = int(i)
                        sm['plate-names'][i - 1] = name
                    except:
                        self._logger.error(
                            "Could not parse plate {0} ({1})".format(i, name))

            #CHECK WHICH PLATES HAVE PLACE IN HISTORY
            if (sm['gridding-history'] is not None and sm['pinnings'] is not None
                    and sm['uuid'] is not None and sm['pinning-formats'] is not None):

                gh = sm['gridding-history']
                sm['gridding-in-history'] = [
                    gh.get_gridding_history_specific_plate(
                        sm['uuid'], p, sm['pinning-formats'][p])
                    for p in xrange(len(sm['pinnings']))]

            return True

        return False


class Analysis_First_Pass(controller_generic.Controller):

    ID_PROJECT = "Project ID"
    ID_LAYOUT = "Layout ID"
    ID_CONTROL = "Control ID"

    def __init__(self, parent, view=None, model=None):

        super(Analysis_First_Pass, self).__init__(
            parent, view=view, model=model)

        self.set_specific_model(model_analysis.copy_model(
            model_analysis.specific_first))

        self._logger = logger.Logger("1st Pass Analysis Controller")

        self._paths = self.get_top_controller().paths

    def start(self, *args, **kwargs):

        """
        sm = self._specific_model

        #Fix the GUI to show it has been started
        view = self.get_view()
        view.get_top().hide_button()
        view.set_stage(view_analysis.Analysis_Stage_First_Pass_Running(
            self, self._model))

        #Calculate path for make project instructions
        p = os.path.join(sm['output-directory'],
                         self._paths.experiment_rebuild_instructions)

        #Prepare instructions for make project processs
        config = ConfigParser()
        config.add_section(Make_Project.CONFIG_META)
        config.add_section(Make_Project.CONFIG_OTHER)
        for key, val in sm.items():
            if key == 'meta-data':
                if 'Version' in val:
                    val['Version'] = __version__
                config.set(Make_Project.CONFIG_META, key, val)
            elif key == 'image-list-model':

                config.set(Make_Project.CONFIG_OTHER,
                           'image-list',
                           [row[0] for row in val])

            else:
                config.set(Make_Project.CONFIG_OTHER, key, val)

        with open(p, 'wb') as configfile:
            config.write(configfile)

        #Register subprocess request
        tc = self.get_top_controller()
        tc.add_subprocess(
            gui_communicator.EXPERIMENT_REBUILD,
            rebuild_instructions_path=p)

        self.set_saved()
        """
    def get_ctrl_id_num(self, projectId, layoutId):

        return verificationTags.ctrlNum(projectId, layoutId)

    def set_output_dir(self, widget):
        m = self._model
        sm = self._specific_model

        dir_list = view_generic.select_dir(
            title=m['analysis-stage-first-dir'],
            start_in=sm['output-directory'])

        if dir_list is not None:

            self.set_unsaved()
            sm['output-directory'] = dir_list
            sm['experiments-root'] = os.sep.join(dir_list.split(os.sep)[:-1])
            sm['experiment-prefix'] = dir_list.split(os.sep)[-1]
            sm['output-file'] = \
                self._paths.experiment_first_pass_analysis_relative.format(
                    sm['experiment-prefix'])

            self._load_meta_from_previous_file()
            self._add_images_in_directory()
            stage = self.get_view().get_stage()

            f_path = self._paths.get_fixture_path('fixture',
                                                  own_path=dir_list)

            self._logger.info("Claimed Local Fixture Path {0}".format(f_path))
            lc = os.path.isfile(f_path)
            stage.update_local_fixture(lc)

            stage.update()
            self._set_allow_run()

    def _set_allow_run(self):

        set_allow_next = self.get_view().get_top().set_allow_next

        sm = self._specific_model
        md = sm['meta-data']
        if len(sm['image-list-model']) == 0:
            set_allow_next(False)
            return
        elif len(sm['output-file']) == 0 or len(sm['output-directory']) == 0:
            set_allow_next(False)
            return
        elif md is None:
            set_allow_next(False)
            return
        elif (md['Pinning Matrices'] is None
                or len(md['Pinning Matrices']) == 0
                or sum([p is None for p in md['Pinning Matrices']]) ==
                len(md['Pinning Matrices'])):

            set_allow_next(False)
            return

        elif md['Prefix'] == '' and sm['use-local-fixture'] is False:

            set_allow_next(False)
            return

        set_allow_next(True)

    def _add_images_in_directory(self):

        sm = self._specific_model
        im_model = sm['image-list-model']
        if im_model is not None:

            extension = '.tiff'
            directory = sm['output-directory']

            list_images = sorted([os.sep.join((directory, f))
                                  for f in os.listdir(directory)
                                  if f.lower().endswith(extension)])

            #CHECK SO NOT DUPING
            for im in list_images:
                in_model = False
                for row in im_model:
                    if im == row[0]:
                        in_model = False
                        break

                if not in_model:

                    im_model.append((im,))

        self._set_allow_run()

    def _load_meta_from_previous_file(self, f_path=None):

        sm = self._specific_model
        if f_path is None:
            f_path = os.path.join(sm['output-directory'], sm['output-file'])
        if os.path.isfile(f_path) is False:
            f_path = os.path.join(
                sm['output-directory'],
                self._paths.experiment_first_pass_analysis_relative.format(
                    sm['output-directory'].split(os.sep)[-1]))

        meta_data = project_log.get_meta_data(f_path)
        pm = meta_data['Pinning Matrices']
        if pm is not None:
            for i, p in enumerate(pm):

                if p is None or p == "None":

                    meta_data['Pinning Matrices'][i] = None

                else:

                    meta_data['Pinning Matrices'][i] = tuple(p)

        sm['meta-data'] = meta_data
        print f_path
        print meta_data
        self._view.get_stage().update()

    def set_new_plates(self, n_plates):

        md = self._specific_model['meta-data']
        if md is not None:

            if md['Pinning Matrices'] is None:
                md['Pinning Matrices'] = [None] * n_plates

            elif len(md['Pinning Matrices']) > n_plates:
                md['Pinning Matrices'] = md['Pinning Matrices'][:n_plates]

            elif len(md['Pinning Matrices']) < n_plates:

                md['Pinning Matrices'] += ([None] *
                                           (n_plates -
                                            len(md['Pinning Matrices'])))

            self.get_view().get_stage().set_pinning()

        self._set_allow_run()

    def set_pinning(self, widget, plate):

        plate -= 1
        row = widget.get_active()
        model = widget.get_model()
        key = model[row][0]
        pm = self._model['pinning-matrices'][key]

        self._specific_model['meta-data']['Pinning Matrices'][plate] = pm

        self._set_allow_run()

    def set_local_fixture(self, widget, *args):

        tc = self.get_top_controller()
        sm = self._specific_model
        local_name = tc.paths.experiment_local_fixturename

        hasFixture = False
        if os.path.isfile(os.path.join(sm['output-directory'],
                                       local_name)):

            f = config_file.Config_File(
                os.path.join(sm['output-directory'], local_name))

            try:
                hasFixture = (float(f.version) >=
                              tc.config.version_oldest_allow_fixture)

            except:

                pass

        if hasFixture:
            self._specific_model['use-local-fixture'] = widget.get_active()
            self._set_allow_run()

        else:

            stage = self.get_view().get_stage()
            stage.update_local_fixture(has_fixture=False)

    def handle_keypress(self, widget, event):

        if view_analysis.gtk.gdk.keyval_name(event.keyval) == "Delete":

            self.get_view().get_stage().delete_selection()

        self._set_allow_run()

    def update_model(self, widget, event, target):

        sm = self._specific_model
        md = self._specific_model['meta-data']

        t = widget.get_text()

        if target == "scanner":
            md['Scanner'] = t
        elif target == "fixture":
            md['Fixture'] = t
        elif target == "desc":
            md['Description'] = t
        elif target == self.ID_PROJECT:
            md['Project ID'] = t
        elif target == self.ID_LAYOUT:
            md['Scanner Layout ID'] = t
        elif target == "output-file":
            sm['output-file'] = t
        elif target == "prefix":
            md['Prefix'] = t
        else:
            print target, t

        self._set_allow_run()


class Analysis_Image_Controller(controller_generic.Controller):

    def __init__(self, parent, view=None, model=None):

        super(Analysis_Image_Controller, self).__init__(
            parent,
            view=view, model=model)

        self._config = parent.get_top_controller().config
        self._specific_model = None
        self._log = None

    def step_back(self, widget, message):

        sm = self._specific_model

        if message == 'select-images':

            sm['image'] = -1
            sm['plate'] = -1
            stage_call = sm['mode']

        elif message == 'previous-image':

            sm['image'] -= 2
            sm['plate'] = -1

            stage_call = 'normalisation'

        elif message == 'previous-plate':

            sm['plate'] -= 2
            stage_call = 'plate'

        elif message == 'normalisation':

            sm['image'] -= 1
            sm['plate'] = -1
            stage_call = 'normalisation'

        self._parent().set_analysis_stage(None, stage_call, sm)

    def execute_fixture(self, widget, data):

        view, specific_model = data
        view.run_lock()
        fixture_path = self.get_top_controller().paths.fixtures

        self.fixture = imageFixture.Image(
            specific_model["fixture-name"],
            image_path=specific_model['images-list-model'][
                specific_model['image']][0],
            fixture_directory=fixture_path)

        thread = threading.Thread(target=self.fixture.threaded)

        thread.start()

        gobject.timeout_add(281, self._parent()._callback, {
            'view': view,
            'view-function': view.set_progress,
            'view-data': None,
            'view-complete': 1.0,
            'complete-function': self.set_grayscale,
            'thread': thread})

    def _get_scale_slice(self, the_slice, flip_coords=False, factor=4):

        data_sorted = zip(*map(sorted, zip(*the_slice)))
        ret = [[factor * p[flip_coords], factor * p[not(flip_coords)]]
               for p in data_sorted]

        return ret

    def setManualNormNewGrayscale(self, grayscaleName):

        if grayscaleName is None:

            self._specific_model['manual-calibration-target'] = list()
            self._specific_model['manual-calibration-grayscaleName'] = None

        else:

            self._specific_model['manual-calibration-target'] = \
                grayscale.getGrayscaleTargets(grayscaleName)
            self._specific_model['manual-calibration-grayscaleName'] = \
                grayscaleName
            self._manualGrayscale()

    def _manualGrayscale(self):

        sm = self._specific_model
        mc = sm['manual-calibration-positions']

        if (mc is None or len(mc) != 1 or
                sm['manual-calibration-grayscaleName'] is None or
                sm['manual-calibration-target'] is None):

            return False

        gs = imageGrayscale.Analyse_Grayscale(
            target_type=sm['manual-calibration-grayscaleName'])
        coords = mc[-1]
        gsIm = sm['image-array'][
            coords[0][1]: coords[1][1],
            coords[0][0]: coords[1][0]]
        gs.get_grayscale(gsIm)
        sm['manual-calibration-values'] = gs.get_source_values()
        if sm['manual-calibration-values'] is not None:
            self.set_auto_grayscale(
                sm['manual-calibration-values'],
                sm['manual-calibration-target'])
            self._view.get_stage().set_measures_from_lists(
                sm['manual-calibration-values'],
                sm['manual-calibration-target'])

    def setManualNormWithGrayScale(self, useGrayscale):

        sm = self._specific_model
        sm['manual-calibration-values'] = list()
        sm['manual-calibration-grayscale'] = useGrayscale
        sm['manual-calibration-positions'] = list()
        stage = self._view.get_stage()
        stage.clear_measures()
        stage.remove_all_patches()

    def updateManualCalibrationValue(self, dtype, row, newValue):

        if row >= 0:
            sm = self._specific_model
            if dtype == 'source':
                sm['manual-calibration-values'][row] = float(newValue)
            else:
                sm['manual-calibration-target'][row] = float(newValue)

            if sm['manual-calibration-grayscale']:
                self._manualGrayscale()

    def get_previously_detected(self, view, specific_model):

        image = specific_model['images-list-model'][
            specific_model['image']][0]

        im_dir = os.sep.join(image.split(os.sep)[:-1])

        image = image.split(os.sep)[-1]

        extension = ".log"

        log_files = [im_dir + os.sep + f for f in os.listdir(im_dir)
                     if f.lower().endswith(extension)]

        data = None

        for f in log_files:

            data = project_log.get_image_from_log_file(f, image)

            if data is not None:

                break

        if data is None:

            view.set_detect_lock(False)

        else:

            i = 0

            plate_coords = dict()

            #Backwards compatible with spell error in early log-files
            gs_i_key = [k for k in data if 'grayscale_in' in k][0]

            while 'plate_{0}_area'.format(i) in data:

                plate_coords[i] = self._get_scale_slice(
                    data['plate_{0}_area'.format(i)],
                    flip_coords=True)

                i += 1

            specific_model['plate-coords'] = plate_coords

            if data[gs_i_key] is not None:
                grayscaleTarget = data[gs_i_key]
            else:
                grayscaleTarget = self.fixture['grayscaleTarget']

            self.set_auto_grayscale(
                data['grayscale_values'],
                grayscaleTarget)

    def set_no_auto_norm(self):

        sm = self._specific_model
        del sm['auto-transpose'][sm['image']]
        self.get_view().get_top().set_allow_next(False)

    def set_fixture(self, view, fixture_name, specific_model):

        specific_model['fixture-name'] = fixture_name

    def set_auto_grayscale(self, grayscaleSource, grayscaleTarget):

        sm = self._specific_model

        """
        if hasattr(self, "fixture"):
            print self.fixture['grayscale_type'], grayscaleSource, grayscaleTarget
            print self.fixture['grayscaleSource'], self.fixture['grayscaleTarget']
        """
        sm['auto-transpose'][sm['image']] = imageBasics.Image_Transpose(
            sourceValues=grayscaleSource,
            targetValues=grayscaleTarget)

        self.get_view().get_top().set_allow_next(True)

    def set_grayscale(self):

        if self.fixture is not None:

            #gs_targets, gs = self.fixture['grayscale']
            gs_target = self.fixture['grayscaleTarget']
            gs_source = self.fixture['grayscaleSource']
            self.set_auto_grayscale(gs_source, gs_target)
            self.get_view().get_stage().set_image()

            pl = self.fixture.get_plates()
            """
            version = self.fixture['fixture']['version']
            if (version is None or
                    version < self._config.version_first_pass_change_1):

                back_compatible = True

            else:

                back_compatible = False
            """
            plate_coords = dict()
            if pl is not None:

                #s_pattern = "plate_{0}_area"

                for i, p in enumerate(pl):

                    plate_coords[i] = p

            self._specific_model['plate-coords'] = plate_coords

        else:

            raise UnDocumented_Error()

    def set_images_has_fixture(self, widget, *args, **kwargs):

        self._specific_model['fixture'] = widget.get_active()

    def toggle_calibration(self, widget, *args, **kwargs):

        view = self.get_view().get_stage()
        val = widget.get_active()
        sm = self._specific_model
        sm['log-only-calibration'] = val
        if val:
            sm['log-previous-file'] = None

        view.set_is_calibration(val)

    def load_previous_log_file(self, widget, view):

        log_file = view_generic.select_file(
            self._model['analysis-stage-log-title'],
            multiple_files=False,
            file_filter=
            self._model['analysis-stage-log-save-file-filter'],
            start_in=self.get_top_controller().paths.experiment_root)

        if len(log_file) > 0:

            log_file = log_file[0]

            try:

                fs = open(log_file, 'r')

            except:

                return

            headers = fs.readline().strip().split("\t")
            fs.close()

            headers = [h[1:-1] for h in headers]
            interests = [h.split(": ") for h in headers if ":" in h]
            compartments = {i[0] for i in interests}
            measures = {i[1] for i in interests}

            view.set_interests(compartments, measures)

            self._specific_model['log-previous-file'] = log_file

            view.set_lock_selection_of_interests(True)
            view.set_previous_log_file(log_file)

        else:

            view.set_lock_selection_of_interests(False)
            view.set_previous_log_file("")

    def set_new_images(self, widget, view, *args, **kwargs):

        image_list = view_generic.select_file(
            self._model['analysis-stage-image-selection-file-dialogue-title'],
            multiple_files=True,
            file_filter=
            self._model['analysis-stage-image-selection-file-filter'],
            start_in=self.get_top_controller().paths.experiment_root)

        treemodel = self._specific_model['images-list-model']

        if len(treemodel) == 0:

            previous_paths = list()

        else:

            previous_paths = [p[0] for p in treemodel if p[0] is not None]

        for im in image_list:

            if im not in previous_paths:

                treemodel.append((im,))

        self._view.get_top().set_allow_next(len(treemodel) > 0)

    def log_compartments(self, widget):

        rows = widget.get_selected_rows()[1]
        self._specific_model['log-interests'][0] = \
            [self._specific_model['log-compartments-default'][r[0]]
             for r in rows]

    def log_measures(self, widget):

        rows = widget.get_selected_rows()[1]
        self._specific_model['log-interests'][1] = \
            [self._specific_model['log-measures-default'][r[0]]
             for r in rows]

    def handle_mpl_keypress(self, event):

        if event.key == "delete":

            if len(self._specific_model['plate-coords']) > 0:

                del self._specific_model['plate-coords'][-1]
                self._view.get_stage().remove_patch()

    def handle_keypress(self, widget, event):

        sm = self._specific_model

        if view_analysis.gtk.gdk.keyval_name(event.keyval) == "Delete":

            if (sm['stage'] == 'image-selection' or
                    sm['stage'] == 'manual-calibration'):

                self._view.get_stage().delete_selection()

    def remove_selection(self, *stuff):

        sm = self._specific_model

        if (sm['stage'] == 'manual-calibration' and
                sm['manual-calibration-grayscale'] is False):

            mcv = sm['manual-calibration-values']

            val = stuff[0]

            for i, calVal in enumerate(mcv):

                if val == str(calVal):

                    del sm['manual-calibration-positions'][i]
                    del mcv[i]

                    if len(mcv) >= 1:

                        self._view.get_top().set_allow_next(True)

                    else:

                        self._view.get_top().set_allow_next(False)

                    return i

        return -1

    def mouse_button_press(self, event, *args, **kwargs):

        if event.xdata is None or event.ydata is None:

            return None

        pos = (event.xdata, event.ydata)
        sm = self._specific_model

        if event.button == 1:

            if sm['stage'] == 'manual-calibration':

                if sm['manual-calibration-positions'] is None:
                    sm['manual-calibration-positions'] = list()

                mc = sm['manual-calibration-positions']

                if len(mc) > 0 and len(mc[-1]) == 1:

                    mc[-1][0] = pos

                else:

                    mc.append([pos])

                stage = self._view.get_stage()
                if sm['manual-calibration-grayscale']:
                    stage.remove_all_patches()
                stage.place_patch_origin(pos)

            elif self._specific_model['stage'] == 'sectioning':

                pc = self._specific_model['plate-coords']

                if len(pc) > 0 and len(pc[-1]) == 1:

                    pc[-1] = pos

                else:

                    pc.append([pos])

                self._view.get_stage().place_patch_origin(pos)

            elif sm['stage'] == 'plate':

                if self._get_inside_selection(pos):

                    sm['selection-move-source'] = pos

                else:

                    if sm['lock-selection'] is not None:

                        self.set_selection(pos=pos)
                        self._view.get_stage().move_patch_origin(pos)

                    else:

                        sm['selection-move-source'] = None
                        self.set_selection(pos=pos, wh=(0, 0))
                        specific_view = self._view.get_stage()
                        specific_view.move_patch_origin(pos)
                        specific_view.move_patch_target(0, 0)
                        sm['selection-drawing'] = True

    def _get_inside_selection(self, pos):

        if (self._specific_model['selection-origin'] is None or
                self._specific_model['selection-size'] is None):

            return False

        s_origin = self._specific_model['selection-origin']
        s_target = [p + s for p, s in zip(
            s_origin,
            self._specific_model['selection-size'])]

        for d in xrange(2):

            if not(s_origin[d] <= pos[d] <= s_target[d]):

                return False

        return True

    def set_selection(self, pos=False, wh=False):

        if pos is not False:
            self._specific_model['selection-origin'] = pos

        if wh is not False:
            self._specific_model['selection-size'] = wh

    def man_detect_mouse_release(self, event, run_analysis=True):

        pos = (event.ydata, event.xdata)

        if None not in pos and event.button == 1:

            stage = self._view.get_stage()
            center, radius = self._calculate_man_selection_circle(posTarget=pos)
            stage.set_man_detect_circle(center, radius)

            if run_analysis:

                sm = self._specific_model
                sm['plate-section-grid-cell'] = \
                    a_wrapper.get_grid_cell_from_array(
                        sm['plate-section-im-array'], center=center,
                        radius=radius,
                        invoke_transform=sm['plate-is-normed'])

                sm['plate-section-features'] = \
                    sm['plate-section-grid-cell'].get_analysis(no_detect=True)

                stage.set_analysis_image()
                self.set_allow_logging()

    def man_detect_mouse_move(self, event):

        self.man_detect_mouse_release(event, run_analysis=False)

    def man_detect_mouse_press(self, event):

        pos = (event.ydata, event.xdata)
        if None not in pos and event.button == 1:

            stage = self._view.get_stage()
            stage.set_man_detect_circle(*self._calculate_man_selection_circle(
                posOrigin=pos, posTarget=pos))

    def _calculate_man_selection_circle(self, posOrigin=None, posTarget=None):

        prevValues = self._specific_model['man-detection']

        if posOrigin is None:
            posOrigin = prevValues[0]
        if posTarget is None:
            posTarget = prevValues[1]

        origo = [(a + b) / 2.0 for a, b in zip(posOrigin, posTarget)]
        radius = sum([abs(a - b) ** 2 for a, b in
                      zip(posOrigin, posTarget)]) ** 0.5 / 2

        prevValues[0] = posOrigin
        prevValues[1] = posTarget

        return origo, radius

    def mouse_button_release(self, event, *args, **kwargs):

        pos = (event.xdata, event.ydata)

        if event.button == 1:

            if self._specific_model['stage'] == 'manual-calibration':

                mc = self._specific_model['manual-calibration-positions']

                if event.xdata is None or event.ydata is None:

                    if len(mc[-1]) == 1:

                        del mc[-1]
                        self._view.get_stage().remove_patch()

                    return None

                if len(mc[-1]) == 1:

                    origin_pos = mc[-1][0]
                    w = pos[0] - origin_pos[0]
                    h = pos[1] - origin_pos[1]
                    mc[-1] = zip(*map(sorted, zip(origin_pos, pos)))

                    self._view.get_stage().move_patch_target(w, h)

                    if self._specific_model['manual-calibration-grayscale']:
                        self._manualGrayscale()
                    else:
                        self.set_manual_calibration_value(mc[-1])

            elif self._specific_model['stage'] == 'sectioning':

                pc = self._specific_model['plate-coords']

                if event.xdata is None or event.ydata is None:

                    if len(pc[-1]) == 1:

                        del pc[-1]
                        self._view.get_stage().remove_patch()

                    self._view.get_stage().set_focus_on_im()
                    return None

                if len(pc[-1]) == 1:

                    origin_pos = pc[-1][0]
                    w = pos[0] - origin_pos[0]
                    h = pos[1] - origin_pos[1]
                    pc[-1] = zip(*map(sorted, zip(origin_pos, pos)))

                    self._view.get_stage().move_patch_target(w, h)

                    if len(pc) > 0:

                        self._view.get_top().set_allow_next(True)

                    else:

                        self._view.get_top().set_allow_next(False)

                    self._view.get_stage().set_focus_on_im()

            elif self._specific_model['stage'] == 'plate':

                sm = self._specific_model
                view = self._view.get_stage()
                view.set_image_sensitivity(False)
                origin_pos = sm['selection-origin']

                if sm['selection-move-source'] is not None:

                    self.set_selection(pos=self._get_new_selection_origin(pos))
                    view.move_patch_origin(sm['selection-origin'])

                elif sm['lock-selection'] is None and origin_pos is not None \
                        and None not in origin_pos and None not in pos:

                    w = pos[0] - origin_pos[0]
                    h = pos[1] - origin_pos[1]

                    view.move_patch_target(w, h)
                    self.set_selection(wh=(w, h))
                    sm['selection-drawing'] = False

                sm['selection-move-source'] = None
                pos1 = sm['selection-origin']
                wh = view.get_selection_size()
                pos2 = [p + s for p, s in zip(pos1, wh)]
                pos1, pos2 = zip(*map(sorted, zip(pos1, pos2)))
                print pos1, pos2
                sm['plate-section-im-array'] = \
                    sm['plate-im-array'][pos1[1]:pos2[1], pos1[0]:pos2[0]]

                sm['plate-section-grid-cell'] = \
                    a_wrapper.get_grid_cell_from_array(
                        sm['plate-section-im-array'], center=None,
                        radius=None,
                        invoke_transform=sm['plate-is-normed'])

                sm['plate-section-features'] = \
                    sm['plate-section-grid-cell'].get_analysis(no_detect=False)

                if sm['plate-section-grid-cell'].get_overshoot_warning():
                    view.set_warning()
                else:
                    view.unset_warning()

                view.set_section_image()
                view.set_man_detect_circle()
                view.set_analysis_image()

                strain = self._log.get_suggested_strain_name(pos1)
                if strain is not None:
                    view.set_strain(strain)

                self.set_allow_logging()
                view.set_image_sensitivity(True)

    def set_allow_logging(self):

        sm = self._specific_model

        self._view.get_stage().set_allow_logging(
            not(sm['plate-section-features'] is None)
            and self._log.get_all_meta_filled())

    def _get_new_selection_origin(self, pos):

        sm = self._specific_model
        if None not in sm['selection-move-source']:
            sel_move = [n - o for n, o in zip(pos, sm['selection-move-source'])]

            new_origin = [o + m for o, m in zip(sm['selection-origin'], sel_move)]
        else:
            new_origin = sm['selection-origin']

        return new_origin

    def set_manual_calibration_value(self, coords):

        if self._specific_model['manual-calibration-values'] is None:

            self._specific_model['manual-calibration-values'] = list()

        mcv = self._specific_model['manual-calibration-values']

        mcv. append(
            self._specific_model['image-array'][
                coords[0][1]: coords[1][1],
                coords[0][0]: coords[1][0]].mean())

        self._view.get_stage().add_measure(mcv[-1], len(mcv))

        if len(mcv) >= 1:

            self._view.get_top().set_allow_next(True)

        else:

            self._view.get_top().set_allow_next(False)

    def mouse_move(self, event, *args, **kwargs):

        sm = self._specific_model
        pos = (event.xdata, event.ydata)

        if event.xdata is None or event.ydata is None:

            return None

        if sm['stage'] == 'manual-calibration':

            mc = sm['manual-calibration-positions']

            if (mc is not None and len(mc) > 0
                    and len(mc[-1]) == 1):

                origin_pos = mc[-1][0]
                w = pos[0] - origin_pos[0]
                h = pos[1] - origin_pos[1]
                self._view.get_stage().move_patch_target(w, h)

        elif sm['stage'] == 'sectioning':

            pc = self._specific_model['plate-coords']

            if len(pc) > 0 and len(pc[-1]) == 1:

                origin_pos = pc[-1][0]
                w = pos[0] - origin_pos[0]
                h = pos[1] - origin_pos[1]
                self._view.get_stage().move_patch_target(w, h)

        elif sm['stage'] == 'plate':

            if sm['selection-move-source'] is not None:

                self._view.get_stage().move_patch_origin(
                    self._get_new_selection_origin(pos))

            elif (sm['lock-selection'] is None and
                    sm['selection-origin'] is not None and
                    sm['selection-drawing'] is True):

                origin_pos = sm['selection-origin']
                w = pos[0] - origin_pos[0]
                h = pos[1] - origin_pos[1]

                self._view.get_stage().move_patch_target(w, h)

    def set_cell(self, widget, type_of_value):

        stage = self._view.get_stage()

        wh = list(stage.get_selection_size())

        if type_of_value == "height":
            try:
                h = int(widget.get_text())
            except:
                return None
        else:
            h = wh[1]

        if type_of_value == "width":
            try:
                w = int(widget.get_text())
            except:
                return None
        else:
            w = wh[0]

        self.set_selection(wh=(w, h))
        stage.move_patch_target(w, h)

    def set_selection_lock(self, widget):

        if widget.get_active():

            self._specific_model['lock-selection'] = \
                self._view.get_stage().get_selection_size()

        else:

            self._specific_model['lock-selection'] = None

        self._view.get_stage().set_allow_selection_size_change(
            self._specific_model['lock-selection'] is None)

    def set_in_log(self, widget, key):

        self._log.set(key, widget)
        self.set_allow_logging()


class Analysis_Transparency_Controller(Analysis_Image_Controller):

    def __init__(self, parent, view=None, model=None):

        super(Analysis_Transparency_Controller, self).__init__(
            parent, view=view, model=model)

    def build_blank_specific_model(self):

        self.set_specific_model(model_analysis.copy_model(
            model_analysis.specific_transparency))


class Analysis_Project_Controller(controller_generic.Controller):

    def __init__(self, parent, view=None, model=None, **kwargs):

        super(Analysis_Project_Controller, self).__init__(
            parent, view=view, model=model)

        self.build_blank_specific_model()

        sm = self._specific_model
        for k in kwargs:

            if k in sm:
                sm[k] = kwargs[k]

        if sm['analysis-project-log_file'] != '':
            gobject.timeout_add(199, self.set_log_file, None,
                                [sm['analysis-project-log_file']])

    def set_abort(self, *args):

        self._parent().set_analysis_stage(None, "about")

    def build_blank_specific_model(self):

        self.set_specific_model(model_analysis.copy_model(
            model_analysis.specific_project))

    def start(self, *args, **kwargs):

        sm = self._specific_model
        tc = self.get_top_controller()

        view = self.get_view()
        view.get_top().hide_button()
        view.set_stage(
            view_analysis.Analysis_Stage_Project_Running(self, self._model))

        a_dict = tc.config.get_default_analysis_query()
        a_dict['-i'] = sm['analysis-project-log_file']
        a_dict['-o'] = sm['analysis-project-output-path']

        if sm['analysis-project-pinnings-active'] != 'file':

            pm = ""

            for plate in sm['analysis-project-pinnings']:

                pm += str(plate).replace(" ", "") + ":"

            pm = pm[:-1]
            a_dict["-m"] = pm

        #tc.add_subprocess(gui_communicator.ANALYSIS, a_dict=a_dict)

    def set_log_file(self, widget, log_files=None):

        if log_files is None:
            log_files = view_generic.select_file(
                self._model['analysis-stage-project-select-log-file-dialog'],
                multiple_files=False, file_filter=
                self._model['analysis-stage-project-select-log-file-filter'],
                start_in=self.get_top_controller().paths.experiment_root)

        if len(log_files) > 0:

            meta_data, images = project_log.get_log_file(
                log_files[0])

            try:
                validProject = (
                    float(meta_data['Version']) >=
                    self.get_top_controller().config.version_oldest_allow_fixture)
            except:
                validProject = False

            sm = self._specific_model
            stage = self._view.get_stage()

            if validProject:

                sm['analysis-project-log_file'] = log_files[0]
                sm['analysis-project-log_file_dir'] = \
                    os.sep.join(log_files[0].split(os.sep)[: -1])

                sm['analysis-project-pinnings-active'] = 'file'

                if 'Pinning Matrices' in meta_data:

                    pinning_matrices = meta_data['Pinning Matrices']

                else:

                    plates = project_log.get_number_of_plates(
                        meta_data=meta_data, images=images)

                    if plates > 0:

                        pinning_matrices = [None] * plates

                    else:

                        pinning_matrices = None

                sm['analysis-project-pinnings-from-file'] = pinning_matrices
                sm['analysis-project-pinnings'] = copy.copy(pinning_matrices)

            else:

                sm['analysis-project-log_file'] = ""
                sm['analysis-project-log_file_dir'] = ""
                sm['analysis-project-pinnings'] = []

            stage.set_valid_log_file(validProject)

            stage.set_log_file()

            stage.set_log_file_data(
                meta_data['Prefix'], meta_data['Description'],
                str(len(images)))

            stage.set_pinning(pinning_matrices)

            self.set_output_dupe()

        self.set_ready_to_run()

    def set_output(self, widget, view, event):

        output_path = widget.get_text()

        sm = self._specific_model

        if event == "exit" and output_path == "":

            output_path = sm['analysis-project-output-default']

        sm['analysis-project-output-path'] = output_path

        if output_path != widget.get_text():

            view.correct_output_path(output_path)

        self.set_output_dupe(output_path, view)
        self.set_ready_to_run()

    def set_output_dupe(self, rel_path=None, view=None):

        sm = self._specific_model

        if view is None:
            view = self._view.get_stage()

        if rel_path is None:
            rel_path = sm['analysis-project-output-path']

        full_path = sm['analysis-project-log_file_dir'] + \
            os.sep + rel_path

        view.set_output_warning(os.path.isdir(full_path))

    def toggle_set_pinning(self, widget, view):

        sm = self._specific_model

        if widget.get_active():

            sm['analysis-project-pinnings-active'] = 'file'
            view.set_pinning(
                sm['analysis-project-pinnings-from-file'],
                sensitive=False)

        else:

            sm['analysis-project-pinnings-active'] = 'gui'
            view.set_pinning(
                sm['analysis-project-pinnings'],
                sensitive=True)

        self.set_ready_to_run()

    def set_pinning(self, widget, plate, *args, **kwargs):

        sm = self._specific_model
        m = self._model

        pinning_txt = widget.get_active_text()

        if pinning_txt in m['pinning-matrices']:
            pinning = m['pinning-matrices'][pinning_txt]
        else:
            pinning = None

        sm['analysis-project-pinnings'][plate - 1] = pinning

        self.set_ready_to_run()

    def set_ready_to_run(self):

        sm = self._specific_model

        if sm['analysis-project-pinnings-active'] == 'file':

            sm_key = 'analysis-project-pinnings-from-file'

        else:

            sm_key = 'analysis-project-pinnings'

        plates_ok = sum([p is not None for p in sm[sm_key]]) > 0

        file_loaded = sm['analysis-project-log_file'] != ""

        self._view.get_top().set_allow_next(file_loaded and plates_ok)


class Analysis_Log_Controller(controller_generic.Controller):

    def __init__(self, parent, general_model, parent_model):

        model = model_analysis.copy_model(model_analysis.specific_log_book)
        self._parent_model = parent_model
        self._general_model = general_model
        self._look_up_coords = list()
        self._look_up_names = list()

        super(Analysis_Log_Controller, self).__init__(
            parent, model=model,
            view=view_analysis.Analysis_Stage_Log(
                self, general_model, model, parent_model))

        self._logger = logger.Logger("Log Book Controller")

        if self._parent_model['log-only-calibration']:
            self._model['calibration-measures'] = True

        if self._parent_model['log-previous-file'] is not None:
            self._load_previous_file_contents()

    def _get_default_view(self):

        view = view_analysis.Analysis_Stage_Log(
            self, self._general_model,
            self._model, self._parent_model)

        return view

    def _load_previous_file_contents(self):

        log_file = self._parent_model['log-previous-file']
        try:

            fs = open(log_file, 'r')

        except:

            return False

        measures = self._model['measures']
        #headers = fs.readline().strip().split("\t")

        firstRow = True
        for data_row in fs:

            if firstRow:
                firstRow = False
            else:
                data_row = data_row.strip().replace("\t", ',')
                data = eval("[{0}]".format(data_row))
                for i, d in enumerate(data):
                    if (isinstance(d, str) and len(d) > 0 and
                            d[0] in ('[', '(') and d[-1] in (']', ')')):

                        try:
                            data[i] = eval(d)
                        except:
                            pass

                measures.append(data)
                self._view.add_data_row(data)

        fs.close()

        return True

    def get_all_meta_filled(self):

        m = self._model
        pm = self._parent_model

        try:

            all_ok = m['plate-names'][pm['image']] is not None and \
                len(m['plate-names'][pm['image']]) == pm['plate'] + 1 and \
                m['current-strain'] is not None

        except:

            return False

        if all_ok and m['calibration-measures']:

            if m['indie-count'] is None:

                all_ok = False

        return all_ok

    def _set_look_up(self, m):

        self._look_up_coords.append(m[4][0])
        self._look_up_names.append(m[5])

    def _set_look_up_strains(self, plate_index=None):

        self._look_up_coords = list()
        self._look_up_names = list()

        measures = self._model['measures']

        if plate_index is not None:

            for m in measures:

                if m[2] == plate_index:

                    self._set_look_up(m)

        if len(self._look_up_coords) == 0:

            for m in measures:

                self._set_look_up(m)

        self._look_up_coords = np.array(self._look_up_coords)

    def get_suggested_strain_name(self, coords):

        if len(self._look_up_names) == 0:

            self._set_look_up_strains(self._parent_model['plate'])

        if self._look_up_coords.size > 0:

            strain_pos = (
                (self._look_up_coords - coords) ** 2).sum(1).argmin()

            if len(self._look_up_names) > strain_pos >= 0:
                strain = self._look_up_names[strain_pos]
                return strain

        return None

    def get_suggested_plate_name(self, index):

        measures = self._model['measures']

        plate_name = None

        for m in measures:

            if m[2] == index:

                plate_name = m[3]

        self._set_look_up_strains(index)

        return plate_name

    def set(self, key, item):

        if key == 'plate':

            image = self._parent_model['image']
            plate = self._parent_model['plate']

            if len(self._model['images']) <= image:
                self._model['images'].append(
                    self._parent_model['images-list-model'][image][0])
                self._model['plate-names'].append(list())

            if len(self._model['plate-names'][image]) <= plate:

                self._model['plate-names'][image].append(item.get_text())

            else:

                self._model['plate-names'][image][plate] = item.get_text()

        elif key == 'strain':

            self._model['current-strain'] = item.get_text()

        elif key == 'indie-count':

            try:
                self._model['indie-count'] = float(item.get_text())

            except:

                self._model['indie-count'] = None

        elif key == 'measures':

            pm = self._parent_model
            m = self._model

            #META INFO
            measures = [
                m['images'][-1],  # pos 0, image-path
                pm['plate-coords'][pm['plate']],  # pos 1 plate coordinates
                pm['plate'],  # pos 2 plate index
                m['plate-names'][pm['image']][pm['plate']],  # pos 3 plate name
                (pm['selection-origin'], pm['selection-size']),  # pos 4 selection coords
                m['current-strain']]  # 5 strain name

            if m['calibration-measures']:

                #SHOULD BE IN RESOURCE
                blob = pm['plate-section-grid-cell'].get_item(
                    "blob").filter_array.astype(np.bool)
                bg = pm['plate-section-grid-cell'].get_item(
                    "background").filter_array
                bg_mean = pm['plate-section-im-array'][bg].mean()
                blob_pixels = pm['plate-section-im-array'][blob] - bg_mean
                blob_pixels[blob_pixels < 0] = 0
                k = np.unique(blob_pixels)
                c = list()
                for v in k:
                    c.append(blob_pixels[blob_pixels == v].size)

                measures += [m['indie-count'], list(k), c]

            else:

                features = pm['plate-section-features']
                self._logger.debug("Saving info {0}".format(features))

                for compartment in pm['log-interests'][0]:

                    if compartment in features.keys():

                        c = features[compartment]

                        for measure in pm['log-interests'][1]:

                            if measure in c.keys():

                                measures.append(c[measure])

                            else:

                                measures.append(None)

            m['measures'].append(measures)

            self._set_look_up_strains(plate_index=pm['plate'])
            self._view.add_data_row(measures)

        else:

            err = Unknown_Log_Request(
                "The key '{0}' not recognized (item {1} lost)".format(
                key, item))

            raise err

    def save_data(self, widget):

        file_name = view_generic.save_file(
            self._general_model['analysis-stage-log-save-dialog'],
            multiple_files=False,
            file_filter=self._general_model['analysis-stage-log-save-file-filter'],
            start_in=self.get_top_controller().paths.experiment_root)

        file_saved = False

        if len(file_name) > 0:

            file_name = file_name[0]

            #Check so endin was filled in
            ext_str = ".csv"
            if ext_str not in file_name or file_name[-4:] != ext_str:
                file_name += ext_str

            try:

                fs = open(file_name, 'r')
                file_exists = True
                fs.close()

            except:

                file_exists = False

            if file_exists:

                file_exists = not(view_generic.overwrite(
                    self._general_model['analysis-stage-log-overwrite'],
                    file_name, self.get_window()))

            if file_exists is False:

                fs = open(file_name, 'w')

                sep = "\t"
                quoute = '"'
                pm = self._parent_model
                m = self._model

                for i, header in enumerate(pm['log-meta-features']):

                    fs.write("{0}{1}{0}{2}".format(quoute, header, sep))

                if m['calibration-measures']:

                    line = sep.join(m['calibration-measure-labels'])
                    fs.write(line)

                else:

                    for i, compartment in enumerate(pm['log-interests'][0]):

                        for j, measure in enumerate(pm['log-interests'][1]):

                            fs.write("{0}{1}: {2}{0}".format(
                                quoute, compartment, measure))

                            if j + 1 != len(pm['log-interests'][1]):

                                fs.write(sep)

                        if i + 1 != len(pm['log-interests'][0]):

                            fs.write(sep)

                fs.write("\n\r")

                for i, measure in enumerate(m['measures']):

                    for j, val in enumerate(measure):

                        if isinstance(val, int) or isinstance(val, float):

                            fs.write(str(val))

                        else:

                            fs.write("{0}{1}{0}".format(quoute, val))

                        if j + 1 != len(measure):

                            fs.write(sep)

                    #if i + 1 != len(m['measures']):

                    fs.write("\n\r")

                fs.close()

                file_saved = True
                self._parent().set_saved()

                view_generic.dialog(
                    self.get_window(),
                    self._general_model['analysis-stage-log-saved'],
                    d_type='info')

        if file_saved is False:

            view_generic.dialog(
                self.get_window(),
                self._general_model['analysis-stage-log-not-saved'],
                d_type='warning')

    def handle_keypress(self, widget, event):

        if view_analysis.gtk.gdk.keyval_name(event.keyval) == "Delete":

            self._view.delete_selection()

    def remove_selection(self, *stuff):

        m = self._model

        im_path = stuff[0]
        plate = int(stuff[1])
        pos = eval(stuff[2])

        for i in xrange(len(m['measures'])):

            if (im_path == m['measures'][i][0] and
                    plate == m['measures'][i][2] and
                    pos == m['measures'][i][4]):

                del m['measures'][i]

                return i

        return -1
