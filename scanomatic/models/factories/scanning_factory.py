import os
import re
import string
from types import StringTypes

from scanomatic.generics.abstract_model_factory import (
    AbstractModelFactory, email_serializer)
from scanomatic.models.scanning_model import (
    ScanningModel, ScannerModel, ScanningAuxInfoModel, PlateDescription,
    CULTURE_SOURCE, PLATE_STORAGE, ScannerOwnerModel)
import scanomatic.io.fixtures as fixtures
import scanomatic.io.app_config as app_config
from scanomatic.models.rpc_job_models import RPCjobModel
from scanomatic.data_processing.calibration import get_active_cccs


class PlateDescriptionFactory(AbstractModelFactory):

    MODEL = PlateDescription
    STORE_SECTION_HEAD = ("name",)
    STORE_SECTION_SERIALIZERS = {
        'name': str,
        'index': int,
        'description': str
    }

    @classmethod
    def create(cls, **settings):
        """
        :rtype : scanomatic.models.scanning_model.PlateDescription
        """
        return super(cls, PlateDescriptionFactory).create(**settings)

    @classmethod
    def _validate_index(cls, model):
        """
        :type model: scanomatic.models.scanning_model.PlateDescription
        """

        if not isinstance(model.index, int):
            return model.FIELD_TYPES.index
        elif model.index >= 0:
            return True
        else:
            return model.FIELD_TYPES.index

    @classmethod
    def _validate_name(cls, model):
        """
        :type model: scanomatic.models.scanning_model.PlateDescription
        """
        if isinstance(model.name, StringTypes) and model.str:
            return True
        return model.FIELD_TYPES.name

    @classmethod
    def _validate_name(cls, model):
        """
        :type model: scanomatic.models.scanning_model.PlateDescription
        """
        if isinstance(model.description, StringTypes) and model.str:
            return True
        return model.FIELD_TYPES.description


class ScanningAuxInfoFactory(AbstractModelFactory):
    MODEL = ScanningAuxInfoModel
    STORE_SECTION_HEAD = "Auxillary Information"
    _SUB_FACTORIES = {

    }
    STORE_SECTION_SERIALIZERS = {
        'stress_level': int,
        'plate_storage': PLATE_STORAGE,
        'plate_age': float,
        'pinning_project_start_delay': float,
        'precultures': int,
        'culture_freshness': int,
        'culture_source': CULTURE_SOURCE
    }

    @classmethod
    def create(cls, **settings):
        """


        :rtype : scanomatic.models.scanning_model.ScanningAuxInfoModel
        """
        return super(cls, ScanningAuxInfoFactory).create(**settings)

    @classmethod
    def _validate_stress_level(cls, model):
        """

        :type model: scanomatic.models.scanning_model.ScanningAuxInfoModel
        """

        if not isinstance(model.stress_level, int):
            return model.FIELD_TYPES.stress_level
        elif model.stress_level is -1 or model.stress_level > 0:
            return True
        else:
            return model.FIELD_TYPES.stress_level

    @classmethod
    def _validate_plate_storage(cls, model):
        """

        :type model: scanomatic.models.scanning_model.ScanningAuxInfoModel
        """
        if cls._is_enum_value(
                model.plate_storage,
                cls.STORE_SECTION_SERIALIZERS[
                    model.FIELD_TYPES.plate_storage.name]):

            return True
        return model.FIELD_TYPES.plate_storage

    @classmethod
    def _validate_plate_age(cls, model):
        """

        :type model: scanomatic.models.scanning_model.ScanningAuxInfoModel
        """
        if (not isinstance(model.plate_age, int) and
                not isinstance(model.plate_age, float)):
            return model.FIELD_TYPES.plate_age
        elif model.plate_age == -1 or model.plate_age > 0:
            return True
        else:
            return model.FIELD_TYPES.plate_age

    @classmethod
    def _validate_pinnig_proj_start_delay(cls, model):
        """

        :type model: scanomatic.models.scanning_model.ScanningAuxInfoModel
        """
        if (not isinstance(model.pinning_project_start_delay, int) and
                not isinstance(model.pinning_project_start_delay, float)):

            return model.FIELD_TYPES.pinning_project_start_delay
        elif (model.pinning_project_start_delay == -1 or
                model.pinning_project_start_delay > 0):
            return True
        else:
            return model.FIELD_TYPES.pinning_project_start_delay

    @classmethod
    def _validate_precultures(cls, model):
        """

        :type model: scanomatic.models.scanning_model.ScanningAuxInfoModel
        """
        if not isinstance(model.precultures, int):
            return model.FIELD_TYPES.precultures
        elif model.precultures == -1 or model.precultures >= 0:
            return True
        else:
            return model.FIELD_TYPES.precultures

    @classmethod
    def _validate_culture_freshness(cls, model):
        """

        :type model: scanomatic.models.scanning_model.ScanningAuxInfoModel
        """
        if not isinstance(model.culture_freshness, int):
            return model.FIELD_TYPES.culture_freshness
        elif model.culture_freshness == -1 or model.culture_freshness > 0:
            return True
        else:
            return model.FIELD_TYPES.culture_freshness

    @classmethod
    def _validate_culture_source(cls, model):
        """

        :type model: scanomatic.models.scanning_model.ScanningAuxInfoModel
        """
        if cls._is_enum_value(model.culture_source, CULTURE_SOURCE):
            return True
        else:
            return model.FIELD_TYPES.culture_source


class ScanningModelFactory(AbstractModelFactory):
    MODEL = ScanningModel
    _GET_MIN_MODEL = app_config.Config().get_min_model
    _GET_MAX_MODEL = app_config.Config().get_max_model
    STORE_SECTION_HEAD = ("project_name",)
    _SUB_FACTORIES = {
        ScanningAuxInfoModel: ScanningAuxInfoFactory,
        PlateDescription: PlateDescriptionFactory
    }

    STORE_SECTION_SERIALIZERS = {
        'start_time': float,
        'number_of_scans': int,
        'time_between_scans': float,
        'project_name': str,
        'directory_containing_project': str,
        'description': str,
        'plate_descriptions': (tuple, PlateDescription),
        'email': email_serializer,
        'pinning_formats': (tuple, tuple, int),
        'fixture': str,
        'scanner': int,
        'scanner_hardware': str,
        'mode': str,
        'computer': str,
        'version': str,
        'id': str,
        'cell_count_calibration_id': str,
        'auxillary_info': ScanningAuxInfoModel,
        'scanning_program': str,
        'scanning_program_version': str,
        'scanning_program_params': (tuple, str)
    }

    @classmethod
    def create(cls, **settings):
        """

        :rtype : scanomatic.models.scanning_model.ScanningModel
        """
        if not settings.get('cell_count_calibration_id', None):

            settings['cell_count_calibration_id'] = 'default'

        return super(cls, ScanningModelFactory).create(**settings)

    @classmethod
    def clamp(cls, model):
        """

        :type model: scanomatic.models.scanning_model.ScanningModel
        """
        return cls._clamp(model, cls._GET_MIN_MODEL(model, factory=cls),
                          cls._GET_MAX_MODEL(model, factory=cls))

    # noinspection PyMethodOverriding
    @classmethod
    def _correct_type_and_in_bounds(cls, model, attr, dtype):
        """

        :type model: scanomatic.models.scanning_model.ScanningModel
        """
        return super(ScanningModelFactory, cls)._correct_type_and_in_bounds(
            model, attr, dtype, cls._GET_MIN_MODEL, cls._GET_MAX_MODEL)

    @classmethod
    def _validate_number_of_scans(cls, model):
        """

        :type model: scanomatic.models.scanning_model.ScanningModel
        """
        return cls._correct_type_and_in_bounds(model, "number_of_scans", int)

    @classmethod
    def _validate_time_between_scans(cls, model):
        """

        :type model: scanomatic.models.scanning_model.ScanningModel
        """
        try:
            model.time_between_scans = float(model.time_between_scans)
        except:
            return model.FIELD_TYPES.time_between_scans

        return cls._correct_type_and_in_bounds(
            model, "time_between_scans", float)

    @classmethod
    def _validate_project_name(cls, model):
        """

        :type model: scanomatic.models.scanning_model.ScanningModel
        """
        if not model.project_name or len(model.project_name) != len(
                tuple(
                    c for c in model.project_name if c in
                    string.letters + string.digits + "_")):

            return model.FIELD_TYPES.project_name

        try:
            int(model.project_name)
            return model.FIELD_TYPES.project_name
        except (ValueError, TypeError):
            pass

        return True

    @classmethod
    def _validate_directory_containing_project(cls, model):
        """

        :type model: scanomatic.models.scanning_model.ScanningModel
        """
        try:

            if os.path.isdir(
                    os.path.abspath(model.directory_containing_project)):
                return True

        except:

            pass

        return model.FIELD_TYPES.directory_containing_project

    @classmethod
    def _validate_description(cls, model):

        """

        :type model: scanomatic.models.scanning_model.ScanningModel
        """

        if isinstance(model.description, StringTypes):
            return True

        return model.FIELD_TYPES.description

    @classmethod
    def _validate_email(cls, model):
        """

        :type model: scanomatic.models.scanning_model.ScanningModel
        """
        if not model.email:
            return True

        if isinstance(model.email, StringTypes):
            email = ",".split(model.email)
        else:
            email = model.email

        try:
            for address in email:

                if (not (
                        isinstance(address, StringTypes) and
                        (address == '' or re.match(
                            r'[^@]+@[^@]+\.[^@]+', address)))):

                    raise TypeError
            return True
        except TypeError:
            return model.FIELD_TYPES.email

    @classmethod
    def _validate_pinning_formats(cls, model):
        """

        :type model: scanomatic.models.scanning_model.ScanningModel
        """
        if AbstractModelFactory._is_pinning_formats(model.pinning_formats):
            return True

        return model.FIELD_TYPES.pinning_formats

    @classmethod
    def _validate_fixture(cls, model):
        """

        :type model: scanomatic.models.scanning_model.ScanningModel
        """
        if model.fixture in fixtures.Fixtures() or not model.fixture:
            return True

        return model.FIELD_TYPES.fixture

    @classmethod
    def _validate_scanner(cls, model):
        """

        :type model: scanomatic.models.scanning_model.ScanningModel
        """
        if app_config.Config().get_scanner_name(model.scanner) is not None:
            return True

        return model.FIELD_TYPES.scanner

    @classmethod
    def _validate_plate_descriptions(cls, model):
        """

        :type model: scanomatic.models.scanning_model.ScanningModel
         """
        if not not isinstance(
                model.plate_descriptions,
                cls.STORE_SECTION_SERIALIZERS[
                    model.FIELD_TYPES.plate_descriptions.name][1]):

            return model.FIELD_TYPES.plate_descriptions

        else:
            for plate_description in model.plate_descriptions:

                if (not isinstance(
                        plate_description,
                        cls.STORE_SECTION_SERIALIZERS[
                            model.FIELD_TYPES.plate_descriptions.name][1])):

                    return model.FIELD_TYPES.plate_descriptions

            if (len(set(plate_description.name for plate_description in
                    model.plate_descriptions)) != len(
                        model.plate_descriptions)):

                return model.FIELD_TYPES.plate_descriptions

        return True

    @classmethod
    def _validate_cell_count_calibration_id(cls, model):
        """

        :type model: scanomatic.models.scanning_model.ScanningModel
        """
        if model.cell_count_calibration_id in get_active_cccs():
            return True
        return model.FIELD_TYPES.cell_count_calibration

    @classmethod
    def _validate_aux_info(cls, model):
        """

        :type model: scanomatic.models.scanning_model.ScanningModel
        """
        if cls._is_valid_submodel(model, "auxillary_info"):
            return True
        return model.FIELD_TYPES.auxillary_info


class ScannerOwnerFactory(AbstractModelFactory):

    MODEL = ScannerOwnerModel
    STORE_SECTION_HEAD = ("id",)

    STORE_SECTION_SERIALIZERS = {
        "id": str,
        "pid": int
    }

    @classmethod
    def create(cls, **settings):
        """
        :rtype : scanomatic.model.scanning_model.ScannerOwnerModel
        """

        return super(ScannerOwnerFactory, cls).create(**settings)


class ScannerFactory(AbstractModelFactory):

    MODEL = ScannerModel
    STORE_SECTION_HEAD = ("scanner_name",)
    _SUB_FACTORIES = {
        ScannerOwnerModel: ScannerOwnerFactory,
        RPCjobModel: ScannerOwnerFactory
    }

    STORE_SECTION_SERIALIZERS = {
        'socket': int,
        'scanner_name': str,
        'usb': str,
        'power': bool,
        "expected_interval": float,
        "email": email_serializer,
        "warned": bool,
        "owner": ScannerOwnerModel,
        "claiming": bool,
        "power": bool,
        "reported": bool,
        "last_on": int,
        "last_off": int,
    }

    @classmethod
    def create(cls, **settings):
        """
         :rtype : scanomatic.models.scanning_model.ScannerModel
        """

        return super(ScannerFactory, cls).create(**settings)
