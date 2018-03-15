"""add scanjobs termination_time and termination_message columns

Revision ID: ffc667adde36
Revises: f9a3e426a0ce
Create Date: 2018-03-15 14:43:16.781596

"""
from __future__ import absolute_import

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'ffc667adde36'
down_revision = 'f9a3e426a0ce'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'scanjobs',
        sa.Column(
            'termination_time', sa.DateTime(timezone=True)
        ),
    )
    op.add_column(
        'scanjobs',
        sa.Column('termination_message', sa.Text()),
    )
    op.create_check_constraint(
        'scanjobs_check_termination_time', 'scanjobs', '''
        termination_time IS NULL
        OR (start_time IS NOT NULL and start_time < termination_time)
        '''
    )


def downgrade():
    pass
