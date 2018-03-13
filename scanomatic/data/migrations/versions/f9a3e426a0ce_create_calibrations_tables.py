"""add calibrations tables

Revision ID: f9a3e426a0ce
Revises: 7c8828cac5fd
Create Date: 2018-03-02 16:37:58.808247

"""
from __future__ import absolute_import

from alembic import op
import sqlalchemy as sa
import sqlalchemy.dialects.postgresql as pg

# revision identifiers, used by Alembic.
revision = 'f9a3e426a0ce'
down_revision = '7c8828cac5fd'
branch_labels = None
depends_on = None


def upgrade():
    calibrations_table = op.create_table(
        'calibrations',
        sa.Column('id', sa.Text(), primary_key=True),
        sa.Column('species', sa.Text(), nullable=False),
        sa.Column('reference', sa.Text(), nullable=False),
        sa.Column(
            'status',
            sa.Enum(
                'under construction', 'active', 'deleted',
                name='calibration_status',
            ),
            nullable=False,
        ),
        sa.Column('polynomial', pg.ARRAY(sa.Float, dimensions=1)),
        sa.Column('edit_access_token', sa.Text()),
        sa.UniqueConstraint(
            'species', 'reference',
            name='uq_species_reference'
        ),
    )
    op.create_table(
        'calibration_images',
        sa.Column('calibration_id', sa.Text(), primary_key=True),
        sa.Column('id', sa.Text(), primary_key=True),
        sa.Column('grayscale_name', sa.Text()),
        sa.Column(
            'grayscale_source_values', pg.ARRAY(sa.Float, dimensions=1)),
        sa.Column(
            'grayscale_target_values', pg.ARRAY(sa.Float, dimensions=1)),
        sa.Column('fixture', sa.Text()),
        sa.Column('marker_x', pg.ARRAY(sa.Float, dimensions=1)),
        sa.Column('marker_y', pg.ARRAY(sa.Float, dimensions=1)),
        sa.ForeignKeyConstraint(
            ['calibration_id'],
            ['calibrations.id'],
            name='fk_calibration_images_calibration_key'
        ),
    )
    op.create_table(
        'calibration_plates',
        sa.Column('calibration_id', sa.Text(), primary_key=True),
        sa.Column('image_id', sa.Text(), primary_key=True),
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('grid_cell_width', sa.Float(), nullable=False),
        sa.Column('grid_cell_height', sa.Float(), nullable=False),
        sa.Column('grid_rows', sa.Integer(), nullable=False),
        sa.Column('grid_cols', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ['calibration_id', 'image_id'],
            ['calibration_images.calibration_id', 'calibration_images.id'],
            name='fk_calibration_plates_image_key',
        ),
    )
    op.create_table(
        'calibration_measurements',
        sa.Column('calibration_id', sa.Text(), primary_key=True),
        sa.Column('image_id', sa.Text(), primary_key=True),
        sa.Column('plate_id', sa.Integer(), primary_key=True),
        sa.Column('row', sa.Integer(), primary_key=True),
        sa.Column('col', sa.Integer(), primary_key=True),
        sa.Column(
            'source_values', pg.ARRAY(sa.Float, dimensions=1), nullable=False),
        sa.Column(
            'source_value_counts', pg.ARRAY(sa.Integer, dimensions=1), nullable=False),
        sa.Column('cell_count', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ['calibration_id', 'image_id', 'plate_id'],
            [
                'calibration_plates.calibration_id',
                'calibration_plates.image_id',
                'calibration_plates.id',
            ],
            name='fk_calibration_measurements_plate_key',
        ),
    )
    op.bulk_insert(calibrations_table, [{
        'id': 'default',
        'species': 'S. cerevisiae',
        'reference': 'Zackrisson et. al. 2016',
        'polynomial': [
            3.379796310880545e-05, 0., 0., 0., 48.99061427688507, 0.
        ],
        'status': 'active',
    }])


def downgrade():
    pass
