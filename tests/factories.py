from __future__ import absolute_import

from scanomatic.io import ccc_data
from scanomatic.io.ccc_data import CalibrationEntryStatus, CellCountCalibration


def make_calibration(
    identifier='ccc000',
    polynomial=[0, 1, 2, 3, 4, 5],
    access_token='password',
    active=False,
):
    ccc = ccc_data.get_empty_ccc_entry(identifier, 'Bogus schmogus', 'Dr Lus')
    if polynomial is not None:
        ccc[CellCountCalibration.polynomial] = (
            ccc_data.get_polynomal_entry(len(polynomial) - 1, polynomial)
        )
    ccc[CellCountCalibration.edit_access_token] = access_token
    if active:
        ccc[CellCountCalibration.status] = CalibrationEntryStatus.Active
    else:
        ccc[CellCountCalibration.status] = (
            CalibrationEntryStatus.UnderConstruction
        )
    return ccc
