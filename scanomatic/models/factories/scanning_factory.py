from scanomatic.generics.abstract_model_factory import AbstractModelFactory
from scanomatic.models.scanning_model import ScanningModel, ScannerOwnerModel
import scanomatic.io.fixtures as fixtures
import scanomatic.io.app_config as app_config

import os
import re
import string


class ScanningModelFactory(AbstractModelFactory):
    _MODEL = ScanningModel
    _GET_MIN_MODEL = app_config.Config().getMinModel
    _GET_MAX_MODEL = app_config.Config().getMaxModel
    STORE_SECTION_HEAD = ("scanner",)
    STORE_SECTION_SERLIALIZERS = {
        ('number_of_scans',): int,
        ('time_between_scans',): float,
        ('project_name',): str,
        ('directory_containing_project',): str,
        ('project_tag',): str,
        ('scanner_tag',): str,
        ('description',): str,
        ('email',): str,
        ('pinning_formats',): tuple,
        ('fixture',): str,
        ('scanner',): int,
        ('mode',): str
    }

    @classmethod
    def clamp(cls, model):

        return cls._clamp(model, cls._GET_MIN_MODEL(model, factory=cls),
                          cls._GET_MAX_MODEL(model, factory=cls))

    # noinspection PyMethodOverriding
    @classmethod
    def _correct_type_and_in_bounds(cls, model, attr, dtype):

        return super(ScanningModelFactory, cls)._correct_type_and_in_bounds(
            model, attr, dtype, cls._GET_MIN_MODEL, cls._GET_MAX_MODEL)

    @classmethod
    def _validate_number_of_scans(cls, model):

        return cls._correct_type_and_in_bounds(model, "number_of_scans", int)

    @classmethod
    def _validate_time_between_scans(cls, model):

        try:
            model.time_between_scans = float(model.time_between_scans)
        except:
            return model.FIELD_TYPES.time_between_scans

        return cls._correct_type_and_in_bounds(model, "time_between_scans", float)

    @classmethod
    def _validate_project_name(cls, model):

        try:
            if os.path.isdir(os.path.join(model.directory_containing_project,
                                          model.project_name)):
                return model.FIELD_TYPES.project_name

        except:

            return model.FIELD_TYPES.project_name

        if len(model.project_name) != len(tuple(c for c in model.project_name
                                                if c in string.letters + string.digits + "_")):
            return model.FIELD_TYPES.project_name

        return True

    @classmethod
    def _validate_directory_containing_project(cls, model):

        try:

            if os.path.isdir(os.path.abspath(model.directory_containing_project)):
                return True

        except:

            pass

        return model.FIELD_TYPES.directory_containing_project

    @classmethod
    def _validate_description(cls, model):

        if isinstance(model.description, str):
            return True

        return model.FIELD_TYPES.description

    @classmethod
    def _validate_email(cls, model):

        if (isinstance(model.email, str) and
                (model.email == '' or
                     re.match(r'[^@]+@[^@]+\.[^@]+', model.email))):
            return True

        return model.FIELD_TYPES.email

    @classmethod
    def _validate_pinning_formats(cls, model):
        if AbstractModelFactory._is_pinning_formats(model.pinning_formats):
            return True

        return model.FIELD_TYPES.pinning_formarts

    @classmethod
    def _validate_fixture(cls, model):

        if model.fixture in fixtures.Fixtures():
            return True

        return model.FIELD_TYPES.fixture

    @classmethod
    def _validate_scanner(cls, model):

        if app_config.Config().get_scanner_name(model.scanner) is not None:
            return True

        return model.FIELD_TYPES.scanner


class ScannerOwnerFactory(AbstractModelFactory):

    _MODEL = ScannerOwnerModel
    STORE_SECTION_HEAD = ("scanner_name",)
    STORE_SECTION_SERLIALIZERS = {
        ('socket',): int,
        ('scanner_name',): str,
        ('job_id',): str,
        ('usb',): str,
        ('power',): bool,
        ("expected_interval",): float,
        ("email",): str,
        ("warned",): bool,
        ("owner_pid"): int,
        ("claiming"): bool

    }