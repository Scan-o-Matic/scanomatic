import os

import pytest

from scanomatic.io.meta_data import MetaData2 as MetaData


class TestMetaDataXLSX:

    DATA_PATH = os.path.join(os.path.dirname(__file__), 'data.xlsx')

    @pytest.mark.parametrize(
        'grids',
        [
            [[8, 12]],
        ]
    )
    def test_load(self, grids):

        md = MetaData(grids, self.DATA_PATH)
        assert md.loaded is True

    @pytest.mark.parametrize(
        'bad_grids',
        [
            [[32, 48]],
            [[8, 12], [8, 12]],
            [[4, 12], [4, 12]],
        ]
    )
    def test_load_fail(self, bad_grids):

        md = MetaData(bad_grids, self.DATA_PATH)
        assert not md.loaded

    def test_getting_data_column(self):

        md = MetaData([[8, 12]], self.DATA_PATH)
        data = md.get_column_index_from_all_plates(-1)
        assert data[0][0][0] == 0.288
        assert data[0][1][0] == 0.276

    def test_getting_strain_info(self):

        md = MetaData([[8, 12]], self.DATA_PATH)
        assert md[0][2][0] == [3, 1, 0.269]
