from __future__ import absolute_import

import pytest

from scanomatic.data.calibrationstore import CalibrationStore
from scanomatic.io.ccc_data import (
    CalibrationEntryStatus, CCCImage, CCCMeasurement, CCCPlate,
    CellCountCalibration, get_empty_ccc_entry, get_polynomal_entry
)


@pytest.fixture
def store(dbconnection, dbmetadata):
    return CalibrationStore(dbconnection, dbmetadata)


def make_calibration(
    identifier='ccc000',
    species='S. Kombuchae',
    reference='Anonymous et al., 2020',
    active=False,
    polynomial=None,
    access_token='password',
):
    ccc = get_empty_ccc_entry(identifier, species, reference)
    if polynomial is not None:
        ccc[CellCountCalibration.polynomial] = (
            get_polynomal_entry(len(polynomial) - 1, polynomial)
        )
    ccc[CellCountCalibration.edit_access_token] = access_token
    if active:
        ccc[CellCountCalibration.status] = CalibrationEntryStatus.Active
    else:
        ccc[CellCountCalibration.status] = (
            CalibrationEntryStatus.UnderConstruction
        )
    return ccc


def make_plate(grid_shape=(16, 24), grid_cell_size=(52.5, 53.1)):
    return {
        CCCPlate.grid_shape: grid_shape,
        CCCPlate.grid_cell_size: grid_cell_size,
    }


@pytest.fixture
def calibration01():
    calibration = get_empty_ccc_entry(
        'ccc001',
        'S. Kombuchae',
        'Anonymous et al., 2020',
    )
    calibration[CellCountCalibration.edit_access_token] = 'authorization001'
    return calibration


@pytest.fixture
def calibration02():
    calibration = get_empty_ccc_entry(
        'ccc002',
        'S. Kefirae',
        'Anonymous et al., 2020',
    )
    calibration[CellCountCalibration.status] = CalibrationEntryStatus.Active
    calibration[CellCountCalibration.polynomial] = (
        get_polynomal_entry(3, [0, 1.1, 2.2, 3.3])
    )
    calibration[CellCountCalibration.edit_access_token] = None
    return calibration


class TestAddCalibration:
    def test_add_under_construction(self, dbconnection, store):
        store.add_calibration(make_calibration(
            identifier='ccc001',
            species='S. Kombuchae',
            reference='Anonymous et al., 2020',
            active=False,
            polynomial=None,
            access_token='authorization001',
        ))
        assert list(dbconnection.execute('''
            SELECT id, species, reference, status, polynomial,
                   edit_access_token
            FROM calibrations
            WHERE id = 'ccc001'
        ''')) == [(
            'ccc001',
            'S. Kombuchae',
            'Anonymous et al., 2020',
            'under construction',
            None,
            'authorization001',
        )]

    def test_add_active_calibration(self, dbconnection, store):
        store.add_calibration(make_calibration(
            identifier='ccc001',
            species='S. Kombuchae',
            reference='Anonymous et al., 2020',
            active=True,
            polynomial=[1, 2, 3],
            access_token='authorization001',
        ))
        assert list(dbconnection.execute('''
            SELECT id, species, reference, status, polynomial,
                   edit_access_token
            FROM calibrations
            WHERE id = 'ccc001'
        ''')) == [(
            'ccc001',
            'S. Kombuchae',
            'Anonymous et al., 2020',
            'active',
            [1, 2, 3],
            'authorization001',
        )]

    def test_add_duplicate_id(self, store):
        store.add_calibration(make_calibration(
            identifier='ccc001',
            species='S. Kombuchae',
            reference='Anonymous et al., 2020',
        ))
        with pytest.raises(CalibrationStore.IntegrityError):
            store.add_calibration(make_calibration(
                identifier='ccc001',
                species='S. Kombuchae',
                reference='Anonymous et al., 2021',
            ))

    def test_add_duplicate_species_and_reference(self, store):
        store.add_calibration(make_calibration(
            identifier='ccc001',
            species='S. Kombuchae',
            reference='Anonymous et al., 2020',
        ))
        with pytest.raises(CalibrationStore.IntegrityError):
            store.add_calibration(make_calibration(
                identifier='ccc002',
                species='S. Kombuchae',
                reference='Anonymous et al., 2020',
            ))


class TestGetCalibrationById:
    def test_get_under_construction_calibration(self, store):
        calibration = make_calibration(
            identifier='ccc001',
            active=False,
        )
        store.add_calibration(calibration)
        assert store.get_calibration_by_id('ccc001') == calibration

    def test_get_active_calibration(self, store):
        calibration = make_calibration(
            identifier='ccc001',
            active=True,
            polynomial=[2, 3, 2],
        )
        store.add_calibration(calibration)
        assert store.get_calibration_by_id('ccc001') == calibration

    def test_get_unknown_id(self, store):
        with pytest.raises(LookupError):
            store.get_calibration_by_id('unknown')


class TestGetAllCalibrations:
    def test_get_all(self, store):
        calibration1 = make_calibration(
            identifier='ccc001', species='S. Kombuchae'
        )
        calibration2 = make_calibration(
            identifier='ccc002', species='S. Kefirae'
        )
        default = make_calibration(
            identifier='default',
            species='S. cerevisiae',
            reference='Zackrisson et. al. 2016',
            polynomial=[
                3.37979631088055e-05, 0.0, 0.0, 0.0, 48.9906142768851, 0.0
            ],
            access_token=None,
            active=True
        )
        store.add_calibration(calibration1)
        store.add_calibration(calibration2)
        assert list(store.get_all_calibrations()) == [
            calibration1, calibration2, default
        ]


class TestHasCalibrationWithId:
    def test_exists(self, store):
        store.add_calibration(make_calibration(identifier='ccc001'))
        assert store.has_calibration_with_id('ccc001')

    def test_doesnt_exists(self, store):
        assert not store.has_calibration_with_id('unknown')


class TestSetCalibrationPolynomial:
    def test_set(self, store, calibration01, dbconnection):
        calibration = make_calibration(identifier='ccc001', polynomial=None)
        store.add_calibration(calibration)
        polynomial = get_polynomal_entry(1, [1, 2])
        store.set_calibration_polynomial('ccc001', polynomial)
        assert list(dbconnection.execute('''
            SELECT polynomial FROM calibrations WHERE id='ccc001'
        ''')) == [([1, 2],)]

    def test_set_to_none(self, store, calibration01, dbconnection):
        store.add_calibration(
            make_calibration(identifier='ccc001', polynomial=[1, 2, 3])
        )
        store.set_calibration_polynomial('ccc001', None)
        assert list(dbconnection.execute('''
            SELECT polynomial FROM calibrations WHERE id='ccc001'
        ''')) == [(None,)]

    def test_set_unknown(self, store):
        with pytest.raises(LookupError):
            store.set_calibration_polynomial('unknown', None)


class TestSetCalibrationStatus:
    def test_activate(self, store, dbconnection):
        store.add_calibration(
            make_calibration(identifier='ccc001', active=False))
        store.set_calibration_status('ccc001', CalibrationEntryStatus.Active)
        assert list(dbconnection.execute('''
            SELECT status FROM calibrations WHERE id='ccc001'
        ''')) == [('active',)]

    def test_delete(self, store, dbconnection):
        store.add_calibration(
            make_calibration(identifier='ccc001', active=False))
        store.set_calibration_status('ccc001', CalibrationEntryStatus.Deleted)
        assert list(dbconnection.execute('''
            SELECT status FROM calibrations WHERE id='ccc001'
        ''')) == [('deleted',)]

    def test_activate_unknown(self, store):
        with pytest.raises(LookupError):
            store.set_calibration_status(
                    'unknown', CalibrationEntryStatus.Active)


class TestAddImageToCalibration:
    def test_add_image(self, store, dbconnection):
        store.add_calibration(make_calibration(identifier='ccc001'))
        store.add_image_to_calibration('ccc001', 'image001')
        assert list(dbconnection.execute('''
            SELECT calibration_id, id FROM calibration_images
        ''')) == [('ccc001', 'image001')]

    def test_add_two_images(self, store, dbconnection):
        store.add_calibration(make_calibration(identifier='ccc001'))
        store.add_image_to_calibration('ccc001', 'image001')
        store.add_image_to_calibration('ccc001', 'image010')
        assert list(dbconnection.execute('''
            SELECT calibration_id, id FROM calibration_images
        ''')) == [('ccc001', 'image001'), ('ccc001', 'image010')]

    def test_unknown_calibration(self, store):
        with pytest.raises(CalibrationStore.IntegrityError):
            store.add_image_to_calibration('ccc001', 'image001')

    def test_duplicate_image_id(self, store):
        store.add_calibration(make_calibration(identifier='ccc001'))
        store.add_image_to_calibration('ccc001', 'image001')
        with pytest.raises(CalibrationStore.IntegrityError):
            store.add_image_to_calibration('ccc001', 'image001')

    def test_add_same_id_different_calibration(self, store, dbconnection):
        store.add_calibration(
                make_calibration(identifier='ccc001', species='x'))
        store.add_calibration(
                make_calibration(identifier='ccc002', species='y'))
        store.add_image_to_calibration('ccc001', 'image001')
        store.add_image_to_calibration('ccc002', 'image001')
        assert list(dbconnection.execute('''
            SELECT calibration_id, id FROM calibration_images
        ''')) == [('ccc001', 'image001'), ('ccc002', 'image001')]


class TestUpdateCalibrationImageWithId:
    def test_update(self, store, dbconnection):
        store.add_calibration(make_calibration(identifier='ccc001'))
        store.add_image_to_calibration('ccc001', 'img001')
        store.update_calibration_image_with_id('ccc001', 'img001', {
            CCCImage.grayscale_name: 'kodak',
            CCCImage.grayscale_source_values: [1, 2, 3],
            CCCImage.grayscale_target_values: [4, 5, 6],
            CCCImage.fixture: 'myFixture',
            CCCImage.marker_x: [0, 0, 2],
            CCCImage.marker_y: [1, 0, 1],
        })
        assert list(dbconnection.execute('''
            SELECT grayscale_name, grayscale_source_values,
                   grayscale_target_values, fixture, marker_x, marker_y
            FROM calibration_images
        ''')) == [
            ('kodak', [1, 2, 3], [4, 5, 6], 'myFixture', [0, 0, 2], [1, 0, 1])
        ]

    def test_unknown(self, store, dbconnection):
        with pytest.raises(LookupError):
            store.update_calibration_image_with_id('ccc001', 'img001', {
                CCCImage.grayscale_name: 'kodak',
            })


class TestCountImagesForCalibration:
    def test_no_images(self, store):
        store.add_calibration(make_calibration(identifier='ccc001'))
        assert store.count_images_for_calibration('ccc001') == 0

    def test_images(self, store):
        store.add_calibration(make_calibration(identifier='ccc001'))
        store.add_image_to_calibration('ccc001', 'image001')
        store.add_image_to_calibration('ccc001', 'image002')
        assert store.count_images_for_calibration('ccc001') == 2

    def test_multiple_calibrations(self, store):
        store.add_calibration(
                make_calibration(identifier='ccc001', species='x'))
        store.add_calibration(
                make_calibration(identifier='ccc002', species='y'))
        store.add_image_to_calibration('ccc001', 'image001')
        store.add_image_to_calibration('ccc001', 'image002')
        store.add_image_to_calibration('ccc002', 'image003')
        assert store.count_images_for_calibration('ccc001') == 2


class TestGetImagesForCalibration:
    def test_get_images(self, store):
        store.add_calibration(
                make_calibration(identifier='ccc001', species='x'))
        store.add_calibration(
                make_calibration(identifier='ccc002', species='y'))
        store.add_image_to_calibration('ccc001', 'image001')
        store.add_image_to_calibration('ccc001', 'image002')
        store.add_image_to_calibration('ccc002', 'image003')
        store.update_calibration_image_with_id('ccc001', 'image001', {
            CCCImage.grayscale_name: 'kodak',
            CCCImage.grayscale_source_values: [1, 2, 3],
            CCCImage.grayscale_target_values: [4, 5, 6],
            CCCImage.fixture: 'myFixture',
            CCCImage.marker_x: [0, 0, 2],
            CCCImage.marker_y: [1, 0, 1],
        })
        assert list(store.get_images_for_calibration('ccc001')) == [
            {
                CCCImage.identifier: 'image001',
                CCCImage.grayscale_name: 'kodak',
                CCCImage.grayscale_source_values: [1, 2, 3],
                CCCImage.grayscale_target_values: [4, 5, 6],
                CCCImage.fixture: 'myFixture',
                CCCImage.marker_x: [0, 0, 2],
                CCCImage.marker_y: [1, 0, 1],
            },
            {
                CCCImage.identifier: 'image002',
                CCCImage.grayscale_name: None,
                CCCImage.grayscale_source_values: None,
                CCCImage.grayscale_target_values: None,
                CCCImage.fixture: None,
                CCCImage.marker_x: None,
                CCCImage.marker_y: None,
            }
        ]


class TestHasCalibrationImageWithId:
    def test_exists(self, store):
        store.add_calibration(make_calibration(identifier='ccc001'))
        store.add_image_to_calibration('ccc001', 'image001')
        assert store.has_calibration_image_with_id('ccc001', 'image001')

    @pytest.mark.parametrize('calibrationid, imageid', [
        ('unknown', 'unknown'),
        ('unknown', 'image001'),
        ('ccc001', 'unknown'),
    ])
    def test_doesnt_exists(self, store, calibrationid, imageid):
        store.add_calibration(make_calibration(identifier='ccc001'))
        store.add_image_to_calibration('ccc001', 'image001')
        assert not store.has_calibration_image_with_id(calibrationid, imageid)


class TestAddPlate:
    def test_add(self, store, dbconnection):
        store.add_calibration(make_calibration(identifier='ccc001'))
        store.add_image_to_calibration('ccc001', 'img001')
        store.add_plate('ccc001', 'img001', 1, make_plate(
            grid_shape=(16, 24), grid_cell_size=(52.5, 53.1),
        ))
        assert list(dbconnection.execute('''
            SELECT calibration_id, image_id, id,
                    grid_cell_height, grid_cell_width, grid_rows, grid_cols
            FROM calibration_plates
        ''')) == [('ccc001', 'img001', 1, 52.5, 53.1, 16, 24)]

    def test_unknown_image(self, store):
        store.add_calibration(make_calibration(identifier='ccc001'))
        with pytest.raises(CalibrationStore.IntegrityError):
            store.add_plate('ccc001', 'img001', 1, make_plate())

    def test_duplicate_plate(self, store):
        store.add_calibration(make_calibration(identifier='ccc001'))
        store.add_image_to_calibration('ccc001', 'img001')
        store.add_plate('ccc001', 'img001', 1, make_plate())
        with pytest.raises(CalibrationStore.IntegrityError):
            store.add_plate('ccc001', 'img001', 1, make_plate())


class TestUpdatePlateWithId:
    def test_update(self, store, dbconnection):
        store.add_calibration(make_calibration(identifier='ccc001'))
        store.add_image_to_calibration('ccc001', 'img001')
        store.add_plate('ccc001', 'img001', 1, make_plate(
            grid_shape=(16, 24), grid_cell_size=(52.5, 53.1),
        ))
        store.add_plate('ccc001', 'img001', 2, make_plate(
            grid_shape=(16, 24), grid_cell_size=(52.5, 53.1),
        ))
        store.update_plate('ccc001', 'img001', 1, {
            CCCPlate.grid_shape: (12, 32),
            CCCPlate.grid_cell_size: (128, 256),
        })
        assert list(dbconnection.execute('''
            SELECT calibration_id, image_id, id,
                    grid_cell_height, grid_cell_width, grid_rows, grid_cols
            FROM calibration_plates
            ORDER BY id
        ''')) == [
            ('ccc001', 'img001', 1, 128, 256, 12, 32),
            ('ccc001', 'img001', 2, 52.5, 53.1, 16, 24),
        ]

    def test_unknown_plate(self, store):
        with pytest.raises(LookupError):
            store.update_plate('unknown', 'unknown', 1, {
                CCCPlate.grid_shape: (12, 32),
                CCCPlate.grid_cell_size: (128, 256),
            })


class TestGetPlateGridCellSize:
    def test_get(self, store):
        store.add_calibration(make_calibration(identifier='ccc001'))
        store.add_image_to_calibration('ccc001', 'img001')
        store.add_plate('ccc001', 'img001', 1, make_plate(
            grid_cell_size=(52.5, 53.1),
        ))
        assert (
            store.get_plate_grid_cell_size('ccc001', 'img001', 1)
            == (52.5, 53.1)
        )

    def test_unknown_plate(self, store):
        with pytest.raises(LookupError):
            store.get_plate_grid_cell_size('ccc001', 'img001', 1)


class TestHasPlateWithId:
    def test_exist(self, store):
        store.add_calibration(make_calibration(identifier='ccc001'))
        store.add_image_to_calibration('ccc001', 'img001')
        store.add_plate('ccc001', 'img001', 1, make_plate())
        assert store.has_plate_with_id('ccc001', 'img001', 1)

    @pytest.mark.parametrize('key', [
        ('ccc001', 'image001', 0),
        ('ccc001', 'unknown', 1),
        ('unknown', 'image001', 1),
    ])
    def test_unknown(self, store, key):
        store.add_calibration(make_calibration(identifier='ccc001'))
        store.add_image_to_calibration('ccc001', 'img001')
        store.add_plate('ccc001', 'img001', 1, make_plate())
        assert not store.has_plate_with_id(*key)


class TestSetMeasurement:
    def test_insert(self, store, dbconnection):
        store.add_calibration(make_calibration(identifier='ccc001'))
        store.add_image_to_calibration('ccc001', 'img001')
        store.add_plate('ccc001', 'img001', 1, make_plate())
        store.set_measurement('ccc001', 'img001', 1, 2, 3, {
            CCCMeasurement.source_values: [4.1, 5.2, 6.3],
            CCCMeasurement.source_value_counts: [7, 8, 9],
            CCCMeasurement.cell_count: 123456,
        })
        assert list(dbconnection.execute('''
            SELECT calibration_id, image_id, plate_id, col, row,
                   source_values, source_value_counts, cell_count
            FROM calibration_measurements
        ''')) == [(
            'ccc001', 'img001', 1, 2, 3, [4.1, 5.2, 6.3], [7, 8, 9], 123456)
        ]

    def test_replace(self, store, dbconnection):
        store.add_calibration(make_calibration(identifier='ccc001'))
        store.add_image_to_calibration('ccc001', 'img001')
        store.add_plate('ccc001', 'img001', 1, make_plate())
        store.set_measurement('ccc001', 'img001', 1, 2, 3, {
            CCCMeasurement.source_values: [4.1, 5.2, 6.3],
            CCCMeasurement.source_value_counts: [7, 8, 9],
            CCCMeasurement.cell_count: 123456,
        })
        store.set_measurement('ccc001', 'img001', 1, 2, 4, {
            CCCMeasurement.source_values: [4.1, 5.2, 6.3],
            CCCMeasurement.source_value_counts: [7, 8, 9],
            CCCMeasurement.cell_count: 123456,
        })
        store.set_measurement('ccc001', 'img001', 1, 2, 3, {
            CCCMeasurement.source_values: [4.4, 5.5, 6.6],
            CCCMeasurement.source_value_counts: [9, 8, 7],
            CCCMeasurement.cell_count: 654321,
        })
        assert list(dbconnection.execute('''
            SELECT calibration_id, image_id, plate_id, col, row,
                   source_values, source_value_counts, cell_count
            FROM calibration_measurements
            ORDER BY row
        ''')) == [
            ('ccc001', 'img001', 1, 2, 3, [4.4, 5.5, 6.6], [9, 8, 7], 654321),
            ('ccc001', 'img001', 1, 2, 4, [4.1, 5.2, 6.3], [7, 8, 9], 123456),
        ]

    def test_unknown_plate(self, store):
        with pytest.raises(CalibrationStore.IntegrityError):
            store.set_measurement('ccc001', 'img001', 1, 2, 3, {
                CCCMeasurement.source_values: [4.1, 5.2, 6.3],
                CCCMeasurement.source_value_counts: [7, 8, 9],
                CCCMeasurement.cell_count: 123456,
            })


class TestHasMeasurementsForPlate:
    def test_exists(self, store):
        store.add_calibration(make_calibration(identifier='ccc001'))
        store.add_image_to_calibration('ccc001', 'img001')
        store.add_plate('ccc001', 'img001', 1, make_plate())
        store.set_measurement('ccc001', 'img001', 1, 2, 3, {
            CCCMeasurement.source_values: [4.1, 5.2, 6.3],
            CCCMeasurement.source_value_counts: [7, 8, 9],
            CCCMeasurement.cell_count: 123456,
        })
        assert store.has_measurements_for_plate('ccc001', 'img001', 1)

    @pytest.mark.parametrize('key', [
        ('unknown', 'img001', 1),
        ('ccc001', 'unknown', 1),
        ('ccc001', 'img001', 0),
    ])
    def test_doesnt_exist(self, store, key):
        store.add_calibration(make_calibration(identifier='ccc001'))
        store.add_image_to_calibration('ccc001', 'img001')
        store.add_plate('ccc001', 'img001', 1, make_plate())
        store.set_measurement('ccc001', 'img001', 1, 2, 3, {
            CCCMeasurement.source_values: [4.1, 5.2, 6.3],
            CCCMeasurement.source_value_counts: [7, 8, 9],
            CCCMeasurement.cell_count: 123456,
        })
        assert not store.has_measurements_for_plate(*key)


class TestGetMeasurementsForCalibration:
    def test_get(self, store):
        store.add_calibration(
                make_calibration(identifier='ccc001', species='x'))
        store.add_calibration(
                make_calibration(identifier='ccc002', species='y'))
        store.add_image_to_calibration('ccc001', 'img001')
        store.add_image_to_calibration('ccc002', 'img002')
        store.add_plate('ccc001', 'img001', 1, make_plate())
        store.add_plate('ccc002', 'img002', 1, make_plate())
        measurement1 = {
            CCCMeasurement.source_values: [4.1, 5.2, 6.3],
            CCCMeasurement.source_value_counts: [7, 8, 9],
            CCCMeasurement.cell_count: 123456,
        }
        measurement2 = {
            CCCMeasurement.source_values: [7.1, 8.2, 9.3],
            CCCMeasurement.source_value_counts: [4, 5, 6],
            CCCMeasurement.cell_count: 4567899,
        }
        measurement3 = {
            CCCMeasurement.source_values: [4.4, 5.5, 6.6],
            CCCMeasurement.source_value_counts: [9, 8, 7],
            CCCMeasurement.cell_count: 654321,
        }
        store.set_measurement('ccc001', 'img001', 1, 1, 1, measurement1)
        store.set_measurement('ccc001', 'img001', 1, 1, 2, measurement2)
        store.set_measurement('ccc002', 'img002', 1, 1, 1, measurement3)
        assert (
            list(store.get_measurements_for_calibration('ccc001'))
            == [measurement1, measurement2]
        )
