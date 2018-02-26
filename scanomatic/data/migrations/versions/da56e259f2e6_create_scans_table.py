"""create scans table

Revision ID: da56e259f2e6
Revises: 39da65401ca3
Create Date: 2018-02-26 18:14:31.166852

"""
from __future__ import absolute_import
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'da56e259f2e6'
down_revision = '39da65401ca3'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'scans',
        sa.Column('id', sa.Text(), primary_key=True),
        sa.Column('start_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('end_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('digest', sa.Text(), nullable=False),
        sa.Column(
            'scanjob_id',
            sa.Text(),
            sa.ForeignKey('scanjobs.id', name='fk_scan_scanjob_id'),
            nullable=False,
        ),
    )


def downgrade():
    pass
