from __future__ import absolute_import

import sqlalchemy as sa
from sqlalchemy.sql.expression import exists, select

from scanomatic.io.ccc_data import (
    CalibrationEntryStatus, CCCImage, CCCMeasurement, CCCPlate, CCCPolynomial,
    CellCountCalibration, get_empty_ccc_entry, get_polynomal_entry
)


def dump_status(status):
    return {
        CalibrationEntryStatus.UnderConstruction: 'under construction',
        CalibrationEntryStatus.Active: 'active',
        CalibrationEntryStatus.Deleted: 'deleted',
    }[status]


class CalibrationStore(object):
    class IntegrityError(Exception):
        pass

    def __init__(self, dbconnection, dbmetadata):
        self._connection = dbconnection
        self._calibrations = dbmetadata.tables['calibrations']
        self._images = dbmetadata.tables['calibration_images']
        self._plates = dbmetadata.tables['calibration_plates']
        self._measurements = dbmetadata.tables['calibration_measurements']

    def add_calibration(self, calibration):
        status = dump_status(calibration[CellCountCalibration.status])
        cccpolynomial = calibration.get(CellCountCalibration.polynomial)
        if cccpolynomial is not None:
            coefficients = cccpolynomial.get(CCCPolynomial.coefficients)
        else:
            coefficients = None
        self._execute(self._calibrations.insert().values(
            id=calibration[CellCountCalibration.identifier],
            species=calibration[CellCountCalibration.species],
            reference=calibration[CellCountCalibration.reference],
            status=status,
            polynomial=coefficients,
            edit_access_token=(
                calibration[CellCountCalibration.edit_access_token]
            ),
        ))

    def get_all_calibrations(self):
        query = self._calibrations.select().order_by(self._calibrations.c.id)
        return self._get_calibrations(query)

    def get_calibration_by_id(self, id_):
        query = self._calibrations.select().where(
            self._calibrations.c.id == id_)
        for calibration in self._get_calibrations(query):
            return calibration
        else:
            raise LookupError(id)

    def _get_calibrations(self, query):
        for row in self._execute(query):
            calibration = get_empty_ccc_entry(
                row['id'], row['species'], row['reference'],
            )
            calibration[CellCountCalibration.edit_access_token] = (
                row['edit_access_token']
            )
            if row['polynomial'] is not None:
                coefficients = row['polynomial']
                power = len(coefficients) - 1
                calibration[CellCountCalibration.polynomial] = (
                    get_polynomal_entry(power, coefficients)
                )
            calibration[CellCountCalibration.status] = {
                'active': CalibrationEntryStatus.Active,
                'under construction': CalibrationEntryStatus.UnderConstruction,
                'deleted': CalibrationEntryStatus.Deleted,
            }[row['status']]
            yield calibration

    def has_calibration_with_id(self, id_):
        return self._exists(
            self._calibrations.select().where(self._calibrations.c.id == id_)
        )

    def set_calibration_polynomial(self, id_, polynomial):
        if polynomial is not None:
            coefficients = polynomial.get(CCCPolynomial.coefficients)
        else:
            coefficients = None
        result = self._execute(
            self._calibrations.update()
            .where(self._calibrations.c.id == id_)
            .values(polynomial=coefficients)
        )
        if result.rowcount == 0:
            raise LookupError(id_)

    def set_calibration_status(self, id_, status):
        result = self._execute(
            self._calibrations.update()
            .where(self._calibrations.c.id == id_)
            .values(status=dump_status(status))
        )
        if result.rowcount == 0:
            raise LookupError(id_)

    def add_image_to_calibration(self, calibrationid, imageid):
        self._execute(self._images.insert().values(
            calibration_id=calibrationid,
            id=imageid,
        ))

    def update_calibration_image_with_id(self, calibrationid, imageid, values):
        columns = {
            CCCImage.grayscale_name: 'grayscale_name',
            CCCImage.grayscale_source_values: 'grayscale_source_values',
            CCCImage.grayscale_target_values: 'grayscale_target_values',
            CCCImage.fixture: 'fixture',
            CCCImage.marker_x: 'marker_x',
            CCCImage.marker_y: 'marker_y',
        }
        values = {columns[key]: value for key, value in values.items()}
        result = self._execute(
            self._images.update()
            .where(sa.and_(
                self._images.c.calibration_id == calibrationid,
                self._images.c.id == imageid,
            ))
            .values(**values)
        )
        if result.rowcount == 0:
            raise LookupError((calibrationid, imageid))

    def count_images_for_calibration(self, calibrationid):
        query = (
            select([sa.func.count()])
            .select_from(self._images)
            .where(self._images.c.calibration_id == calibrationid)
        )
        return self._execute(query).scalar()

    def get_images_for_calibration(self, calibrationid):
        query = (
            self._images.select()
            .where(self._images.c.calibration_id == calibrationid)
        )
        for row in self._execute(query):
            yield {
                CCCImage.identifier: row['id'],
                CCCImage.grayscale_name: row['grayscale_name'],
                CCCImage.grayscale_source_values:
                    row['grayscale_source_values'],
                CCCImage.grayscale_target_values:
                    row['grayscale_target_values'],
                CCCImage.fixture: row['fixture'],
                CCCImage.marker_x: row['marker_x'],
                CCCImage.marker_y: row['marker_y'],
            }

    def has_calibration_image_with_id(self, calibrationid, imageid):
        return self._exists(
            self._images.select().where(
                sa.and_(
                    self._images.c.id == imageid,
                    self._images.c.calibration_id == calibrationid
                )
            )
        )

    def add_plate(self, calibrationid, imageid, plateid, plate):
        self._execute(self._plates.insert().values(
            calibration_id=calibrationid,
            image_id=imageid,
            id=plateid,
            grid_cell_height=plate[CCCPlate.grid_cell_size][0],
            grid_cell_width=plate[CCCPlate.grid_cell_size][1],
            grid_rows=plate[CCCPlate.grid_shape][0],
            grid_cols=plate[CCCPlate.grid_shape][1],
        ))

    def update_plate(self, calibrationid, imageid, plateid, plate):
        result = self._execute(
            self._plates.update()
            .where(sa.and_(
                self._plates.c.id == plateid,
                self._plates.c.calibration_id == calibrationid,
                self._plates.c.image_id == imageid,
            ))
            .values(
                grid_cell_height=plate[CCCPlate.grid_cell_size][0],
                grid_cell_width=plate[CCCPlate.grid_cell_size][1],
                grid_rows=plate[CCCPlate.grid_shape][0],
                grid_cols=plate[CCCPlate.grid_shape][1],
            )
        )
        if result.rowcount == 0:
            raise LookupError((calibrationid, imageid, plateid))

    def get_plate_grid_cell_size(self, calibrationid, imageid, plateid):
        query = (
            self._plates.select()
            .where(sa.and_(
                self._plates.c.id == plateid,
                self._plates.c.calibration_id == calibrationid,
                self._plates.c.image_id == imageid,
            ))
        )
        for row in self._execute(query):
            return (row['grid_cell_height'], row['grid_cell_width'])
        raise LookupError((calibrationid, imageid, plateid))

    def has_plate_with_id(self, calibrationid, imageid, plateid):
        return self._exists(
            self._plates.select().where(
                sa.and_(
                    self._plates.c.id == plateid,
                    self._plates.c.image_id == imageid,
                    self._plates.c.calibration_id == calibrationid,
                )
            )
        )

    def set_measurement(
        self, calibrationid, imageid, plateid, col, row, measurement
    ):
        where = sa.and_(
            self._measurements.c.calibration_id == calibrationid,
            self._measurements.c.image_id == imageid,
            self._measurements.c.plate_id == plateid,
            self._measurements.c.row == row,
            self._measurements.c.col == col,
        )
        if self._exists(self._measurements.select().where(where)):
            self._execute(self._measurements.update().where(where).values(
                source_values=measurement[CCCMeasurement.source_values],
                source_value_counts=(
                    measurement[CCCMeasurement.source_value_counts]
                ),
                cell_count=measurement[CCCMeasurement.cell_count],
            ))
        else:
            self._execute(self._measurements.insert().values(
                calibration_id=calibrationid,
                image_id=imageid,
                plate_id=plateid,
                row=row,
                col=col,
                source_values=measurement[CCCMeasurement.source_values],
                source_value_counts=(
                    measurement[CCCMeasurement.source_value_counts]
                ),
                cell_count=measurement[CCCMeasurement.cell_count],
            ))

    def has_measurements_for_plate(self, calibrationid, imageid, plateid):
        return self._exists(self._measurements.select().where(
            sa.and_(
                self._measurements.c.calibration_id == calibrationid,
                self._measurements.c.image_id == imageid,
                self._measurements.c.plate_id == plateid,
            )
        ))

    def get_measurements_for_calibration(self, calibrationid):
        query = self._measurements.select().where(
            self._measurements.c.calibration_id == calibrationid
        ).order_by(
            self._measurements.c.image_id, self._measurements.c.plate_id,
            self._measurements.c.col, self._measurements.c.row
        )
        for row in self._execute(query):
            yield {
                CCCMeasurement.source_values: row['source_values'],
                CCCMeasurement.source_value_counts: row['source_value_counts'],
                CCCMeasurement.cell_count: row['cell_count'],
            }

    def _execute(self, query):
        try:
            return self._connection.execute(query)
        except sa.exc.IntegrityError as e:
            raise self.IntegrityError(e)

    def _exists(self, query):
        return self._connection.execute(exists(query).select()).scalar()
