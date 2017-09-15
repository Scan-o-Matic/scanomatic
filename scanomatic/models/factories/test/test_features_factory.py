import pytest
from scanomatic.models.factories.features_factory import FeaturesFactory


class TestFeaturesFactory:

    def test_default_will_overwrite_qc(self):

        m = FeaturesFactory.create()
        assert m.try_keep_qc is False

    def test_can_try_keep_qc(self):

        m = FeaturesFactory.create(try_keep_qc=True)
        assert m.try_keep_qc

    def test_try_keep_qc_setting_in_dict_form(self):

        m = FeaturesFactory.create(try_keep_qc=True)
        assert FeaturesFactory.to_dict(m).get('try_keep_qc')
