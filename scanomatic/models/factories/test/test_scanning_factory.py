import pytest

from scanomatic.models.factories.scanning_factory import ScanningModelFactory


class TestCreatingScanningModel:

    def test_creating_valid_minimal_model(self):

        model = ScanningModelFactory.create(project_name='Test')
        assert ScanningModelFactory.validate(model)
        assert model.project_name == 'Test'

    def test_create_valid_minimal_model_with_deprecated_fields(self):

        model = ScanningModelFactory.create(
            project_name='Tests',
            scanner_tag='are',
            project_tag='annoying')

        assert ScanningModelFactory.validate(model)

        with pytest.raises(AttributeError):

            model.scanner_tag

        with pytest.raises(AttributeError):

            model.project_tag

    def test_model_without_project_name_doesnt_validate(self):

        model = ScanningModelFactory.create()

        assert ScanningModelFactory.validate(model) is False
